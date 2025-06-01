import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import numbers
from torch.nn import init


class PositionalEncoding(nn.Module):
    """positional encoding layer, add position information to the sequence"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LongTermEncoder(nn.Module):
    """
    3 DSTCL Layer
    init: c_in - in_dim   c_out - residual_dim
    input : x ∈ [B, C, T_long, N]  T_long
    output :  x ∈ [B, C_out, T_short, N]  T_long->T_short
    """

    def __init__(
        self,
        gcn_true,
        buildA_true,
        gcn_depth,
        num_nodes,
        device,
        T_short,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=8,
        seq_length=12,
        in_dim=2,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
    ):
        super(LongTermEncoder, self).__init__()
        self.gcn_true = gcn_true  # use gcn
        self.buildA_true = buildA_true  # use dynamic adjacency matrix
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A  # predefined adjacency matrix
        self.filter_convs = nn.ModuleList()  #
        self.gate_convs = nn.ModuleList()  #
        self.residual_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )  #
        self.time_proj = nn.AdaptiveAvgPool2d((T_short, num_nodes))
        self.gc = graph_constructor(
            num_nodes,
            subgraph_size,
            node_dim,
            device,
            alpha=tanhalpha,
            static_feat=static_feat,
        )
        self.seq_length = seq_length  #
        kernel_size = 7  #
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential**layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1
                    + i
                    * (kernel_size - 1)
                    * (dilation_exponential**layers - 1)
                    / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i
                        + (kernel_size - 1)
                        * (dilation_exponential**j - 1)
                        / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.gate_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )
                    self.gconv2.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                num_nodes,
                                self.seq_length - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )
                else:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                num_nodes,
                                self.receptive_field - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )

                new_dilation *= dilation_exponential

        self.layers = layers
        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, env, idx=None):
        """
        residual_nums=D_long
        input shape:[batch_size, C, num_nodes,T_long]
        output shape:[batch_size, D_long, T_long, num_nodes]
        """
        seq_len = input.size(3)

        assert (
            seq_len == self.seq_length
        ), "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field - self.seq_length, 0, 0, 0)
            )

        cooldowns = env.cooldowns

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)  #
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

            if cooldowns is not None:
                rc = torch.tensor(cooldowns).float().to(input.device)

                mask_vec = 1.0 - rc  # [N]
                #  [N,N]
                if mask_vec.dim() == 2:  # 
                    mask_mat = mask_vec.unsqueeze(-1) * mask_vec.unsqueeze(
                        -2
                    )  # [B,N,N]
                else:   
                    mask_mat = mask_vec.unsqueeze(-1) * mask_vec.unsqueeze(-2)  # [N,N]
                adp = adp * mask_mat
                adp = F.softmax(adp, dim=-1)

        x = self.start_conv(input)

        for i in range(self.layers):  
            residual = x
            filter = self.filter_convs[i](x)

            filter = torch.tanh(filter)

            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)

            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](
                    x, adp.transpose(1, 0)
                )  
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)
        x = x.permute(0, 1, 3, 2)
        x = self.time_proj(x)
        return x


class ShortTermEncoder(nn.Module):
    """
    Short-term Information Extraction Branch
    init: c_in, d_short:output channel, kernel_size, num_layers, num_nodes
    input: x [B, C, T, N]
    output: [B, D_short, T_short, N]
    """

    def __init__(self, c_in, d_short, kernel_size=2, num_layers=2, num_nodes=225):
        super().__init__()

        self.input_proj = nn.Conv2d(1, d_short, kernel_size=1)

        self.temporal_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        d_short,
                        d_short,
                        (kernel_size, 1),
                        padding=(kernel_size // 2, 0),
                    ),
                    nn.BatchNorm2d(d_short),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_layers)
            ]
        )

        self.spatial_conv = nn.Conv2d(
            d_short,
            d_short,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )

        self.norm = nn.LayerNorm(d_short)

    def forward(self, x):
        # x: [B, C, T, N]
        x = x[:, 0:1, :, :]
        x = self.input_proj(x)  # → [B, D_short, T, N]

        for conv in self.temporal_convs:
            x = x + conv(x) 

        x = self.spatial_conv(x)  # → [B, D_short, T, N]

        # LayerNorm
        x = x.transpose(1, 3)  # → [B, N, T, D_short]
        x = self.norm(x)  
        x = x.transpose(1, 3)  # → [B, D_short, T, N]

        return x


class UnifiedResourceModule(nn.Module):
    """
    unified resource module, handle feature fusion and step prediction
    follow the principle: the richer the resources, the longer the prediction step, the more the long-term feature fusion
    """

    def __init__(
        self,
        hidden_dim,
        num_nodes,
        total_resources,
        max_steps=10,
        num_heads=4,
        dropout=0.1,
    ):
        super(UnifiedResourceModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.total_resources = total_resources
        self.max_steps = max_steps
        self.num_nodes = num_nodes

        self.resource_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )

        self.fusion_weight = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        #
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, short_term, long_term, available_resources):
        """
        Args:
            short_term: [seq_len, batch_size*num_nodes, hidden_dim]
            long_term: [seq_len, batch_size*num_nodes, hidden_dim]
            available_resources: [batch_size] 

        Returns:
            fused_features: [seq_len, batch_size, hidden_dim]
            k: [batch_size] 
            fusion_ratio: [batch_size] 
        """
        device = short_term.device

        seq_len, batch_nodes, hidden_dim = short_term.shape
        batch_size = batch_nodes // self.num_nodes

        if isinstance(available_resources, np.ndarray):
            available_resources = (
                torch.from_numpy(available_resources).float().to(device)
            )
        elif isinstance(available_resources, list):
            available_resources = torch.tensor(
                available_resources, dtype=torch.float32, device=device
            )
        elif isinstance(available_resources, torch.Tensor):
            available_resources = available_resources.to(
                dtype=torch.float32, device=device
            )
        else:
            raise ValueError(
                f"available_resources must be a list, numpy array, or torch.Tensor, but got {type(available_resources)}"
            )
        k_ratio = available_resources / self.total_resources
        expanded_resources = available_resources.repeat_interleave(self.num_nodes)

        # Step 1  resource_ratio ∈ [batch_size]
        resource_ratio = expanded_resources / self.total_resources
        resource_ratio = torch.clamp(resource_ratio, min=0.001, max=1.0)

        # Step 2
        fusion_ratio = resource_ratio.view(1, batch_nodes, 1)  # shape: [1, B*N, 1]

        # 
        fused_features = (
            1 - fusion_ratio
        ) * short_term + fusion_ratio * long_term  # shape: [seq_len, batch_size*num_nodes, hidden_dim]

        # Step 3 
        attn_output, _ = self.multihead_attn(
            fused_features, fused_features, fused_features
        )
        fused_features = self.norm1(fused_features + self.dropout(attn_output))

        # Step 4
        ff_output = self.feedforward(fused_features)
        fused_features = self.norm2(
            fused_features + self.dropout(ff_output)
        )  # shape: [seq_len, batch_size, hidden_dim]

        # Step 5:  k ∈ [batch_size]
        k = torch.clamp(
            (k_ratio * self.max_steps).round(), min=1, max=self.max_steps
        ).long()

        return (
            fused_features,
            k,
            k_ratio,
        )  # shape: [seq_len, batch_size, hidden_dim], [batch_size], [batch_size]


class MultiStepDecoder(nn.Module):
    """multi-step decoder, generate future K step predictions"""

    def __init__(self, hidden_dim, output_dim, max_steps=10, num_heads=4, dropout=0.1):
        super(MultiStepDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_steps = max_steps

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=2
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.step_embedding = nn.Embedding(max_steps + 1, hidden_dim)

    def forward(self, memory, k):
        """
        Args:
            memory: [seq_len, B, hidden_dim]
            k: int
        Returns:
            outputs: list of [B, output_dim], len = k
        """
        B = memory.size(1)
        device = memory.device

        #  [k, B, hidden_dim]
        step_indices = torch.arange(k, device=device)
        step_embeddings = self.step_embedding(step_indices)  # [k, hidden_dim]
        step_embeddings = step_embeddings.unsqueeze(1).expand(
            -1, B, -1
        )  # [k, B, hidden_dim]

        #  decoder_inputs: [k, B, hidden_dim]
        decoder_input = torch.cat(
            [self.start_token.expand(1, B, -1), step_embeddings[:-1]], dim=0
        )

        
        decoder_input = self.pos_encoder(decoder_input)
        tgt_mask = self._generate_square_subsequent_mask(k).to(device)

        decoder_output = self.transformer_decoder(
            decoder_input, memory, tgt_mask=tgt_mask
        )

        output_projected = self.output_projection(decoder_output)
        output_activated = torch.sigmoid(output_projected)

        return [output_activated[i] for i in range(k)]

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask


# Layers


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 4, 6, 8]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor))
            )

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3) :]
        x = torch.cat(x, dim=1)
        return x


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(
                input,
                tuple(input.shape[1:]),
                self.weight[:, idx, :],
                self.bias[:, idx, :],
                self.eps,
            )
        else:
            return F.layer_norm(
                input, tuple(input.shape[1:]), self.weight, self.bias, self.eps
            )

    def extra_repr(self):
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncwl,vw->ncvl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

    def forward(self, x):
        return self.mlp(x)
