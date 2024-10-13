import torch
import argparse


# Model Profiles


def get_parser():
    parser = argparse.ArgumentParser(description='Example Argument Parser')
    parser.add_argument('device', type=str, help='Device to use for training (cuda or cpu)', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('batch_size', type=int, help='Batch size for training', default=4)
    parser.add_argument('max_epoach_no_improve', type=int, help='Number of epochs without improvement to stop training', default=5)
    parser.add_argument('learning_rate', type=float, help='Learning rate for optimizer', default=0.0001)
    parser.add_argument('input_h', type=int, help='Height of the input images', default=384)
    parser.add_argument('input_w', type=int, help='Width of the input images', default=512)
    parser.add_argument('gauss', type=int, help='Width of the input images', default=7)
    parser.add_argument('kpt_n', type=int, help='Number of keypoints', default=6)
    parser.add_argument('hip_classes', type=int, help='Number of hip classes', default=8)
    parser.add_argument('n_expert', type=int, help='Number of experts', default=3)
    parser.add_argument('n_task', type=int, help='Number of tasks', default=2)
    parser.add_argument('--use_gate', action="store_true", help='Whether to use ME-GCT')
    parser.add_argument('--show_point_on_picture', action="store_true", help='Whether or not to visualize the results of coordinate prediction') 

    return parser.parse_args(['cuda', '4', '5', '0.0001', '384', '512','7', '6', '8', '3', '2', '--use_gate'])