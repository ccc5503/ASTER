import argparse
from datetime import datetime
from utils.logger import Logger
from utils.data_loader import AccidentDataset
from torch.utils.data import DataLoader
import yaml

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Train ESTF model')
    parser.add_argument('--config', type=str, default='config.yaml', help='PATH')
    parser.add_argument('--log_dir', type=str, default='logs/train', help='Directory to save logs')
    return parser.parse_args()


def init_logger(log_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = Logger(f'train_{timestamp}', log_dir)
    logger.info(f"Start: {logger.log_file}")
    return logger


def load_training_config(config_path, logger):
    logger.info(f"load_config: {config_path}")
    return load_config(config_path)


def prepare_data(config):
    dataset = AccidentDataset(
        config['data']['data_file'],
        T_long=config['data']['T_long'],
        T_short=config['data']['T_short'],
        K=config['environment']['max_prediction_steps'],
        num_nodes=config['data']['num_nodes'],
        in_dim=config['model']['C'],
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    return dataset, dataloader


def extract_config_params(config):
    training_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    
    return (
        training_cfg['batch_size'],
        training_cfg['num_epochs'],
        training_cfg['learning_rate'],
        training_cfg['gamma'],
        training_cfg['epsilon_start'],
        training_cfg['epsilon_decay'],
        training_cfg['epsilon_min'],
        training_cfg['target_update_freq'],
        training_cfg['total_resources'],
        training_cfg['cooldown_steps'],
        training_cfg['k_steps'],
        data_cfg['T_short'],
        data_cfg['T_long'],
        data_cfg['num_nodes'],
        data_cfg['graph_file'],
        model_cfg['C'],
        model_cfg['D_short'],
        model_cfg['D_long'],
        model_cfg['C_common'],
        model_cfg['out_channels'],
        model_cfg['num_actions_per_node']
    )
