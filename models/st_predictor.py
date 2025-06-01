from .layer import *

class ResourceAwareDualEncoderDecoder(nn.Module):
    """
    resource-aware dual encoder, dynamically fuse long and short term features based on available resources,
    give a suitable prediction step K, then generate K step predictions through the decoder
    """
    def __init__(self, config):
        super(ResourceAwareDualEncoderDecoder, self).__init__()

        self.C = config['model']['C']  # 
        self.D_short = config['model']['D_short']  
        self.D_long = config['model']['D_long']  
        self.C_common = config['model']['C_common']  
        self.out_channels = config['model']['out_channels']  
        self.max_steps = config['environment']['max_prediction_steps'] 
        self.total_resources = config['training']['total_resources']  
        self.num_nodes = config['data']['num_nodes']  
        self.device = config['environment']['device']  
        self.T_long= config['data']['T_long']  
        self.T_short= config['data']['T_short']  
        
        # short term encoder
        self.short_term_encoder = ShortTermEncoder(
            1, 
            self.D_short, 
            kernel_size=3, 
            num_layers=2
        )
        
        # long term encoder
        self.long_term_encoder = LongTermEncoder(
            gcn_true=True,                     
            buildA_true=True,                
            gcn_depth=2,                      
            num_nodes=self.num_nodes,       
            device=self.device,              
            T_short=self.T_short,               
            predefined_A=None,               
            static_feat=None,                
            dropout=0.3,                     
            subgraph_size=20,                
            node_dim=225,                    
            dilation_exponential=2,          
            conv_channels=32,               
            residual_channels=self.D_long,   
            seq_length=self.T_long,          
            in_dim=self.C,                   
            layers=3,                       
            propalpha=0.05,                  
            tanhalpha=3,                    
            layer_norm_affline=True       
        )
        
        # projection layers
        self.project_short = nn.Conv2d(self.D_short, self.C_common, kernel_size=1)
        self.project_long = nn.Conv2d(self.D_long, self.C_common, kernel_size=1)
        
        # resource-aware unified feature fusion module
        self.unified_resource_module = UnifiedResourceModule(
            hidden_dim=self.C_common,
            num_nodes=self.num_nodes,
            total_resources=self.total_resources,
            max_steps=self.max_steps,
            num_heads=2,
            dropout=0.1
        )
        
        # multi-step decoder
        self.decoder = MultiStepDecoder(
            hidden_dim=self.C_common,
            output_dim=self.out_channels,
            max_steps=self.max_steps,
            num_heads=4,
            dropout=0.1
        )
        self.is_ddp = False

    def forward(self, short_term_input, long_term_input, available_resources,env):
        """
        Args:
            short_term_input: [b T_short, num_nodes, C]
            long_term_input: [b T_long, num_nodes, C]
            available_resources: [batch_size]
        
        Returns:
            predictions: list of [batch_size, num_nodes, out_channels]
            hidden_states: [batch_size, num_nodes, C_common]
            k: [batch_size] 
        """
        batch_size,_, num_nodes, _ = short_term_input.size()

        st_in = short_term_input.permute(0, 3, 1, 2)  # output: [batch_size, C, T_short, num_nodes]
        lt_in = long_term_input.permute(0, 3, 2, 1)  # shape: [batch_size, C, num_nodes,T_long]

        st_out = self.short_term_encoder(st_in) # shape: [batch_size, D_short, T_short, num_nodes]
        lt_out = self.long_term_encoder(lt_in,env) # shape: [batch_size, D_long, T_long, num_nodes]

        h_st = self.project_short(st_out) # shape: [batch_size, C_common, T_short, num_nodes]
        h_lt = self.project_long(lt_out)  # shape: [batch_size, C_common, T_short, num_nodes]

        #  -> Transformer  [T, B*N, C]
        h_st_trans = h_st.permute(2, 0, 3, 1).reshape(h_st.size(2), batch_size * num_nodes, self.C_common) # shape: [T_short, B*N, C_common]
        h_lt_trans = h_lt.permute(2, 0, 3, 1).reshape(h_lt.size(2), batch_size * num_nodes, self.C_common) # shape: [T_short, B*N, C_common]

        # resource-aware unified feature fusion
        fused_features, k, fusion_ratio = self.unified_resource_module(
            short_term=h_st_trans,
            long_term=h_lt_trans,
            available_resources=available_resources
        )  # fused_features: [T, B*N, C], k: [B], fusion_ratio: [B]

        # 
        hidden_states = fused_features[-1].view(batch_size, num_nodes, self.C_common)  # shape: [batch_size, num_nodes, C_common]

        # 
        max_k = self.max_steps
        outputs = self.decoder(fused_features, max_k)  #  [B*N, out_channels]

        # 
        predictions = []
        for step in range(max_k):
            step_output = outputs[step].view(batch_size, num_nodes, self.out_channels)
            # 
            mask = (k > step).view(batch_size, 1, 1).expand(-1, num_nodes, self.out_channels)
            step_output = step_output * mask  # 
            predictions.append(step_output)

        return predictions, hidden_states, k   # shape: [k, B, N, out_channels], [B, N, C_common], [B]