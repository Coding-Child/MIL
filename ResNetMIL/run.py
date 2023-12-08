import argparse
from runfile.main import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters for the model.')
    parser.add_argument('--lr', type=float, default=1e-10, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--num_classes', type=int, default=2, help='Numer of Data\'s class.')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Test batch size.')
    parser.add_argument('--num_train_instance', type=int, default=32, help='Training Instance size.')
    parser.add_argument('--num_val_instance', type=int, default=32, help='Validation Instance size.')
    parser.add_argument('--num_test_instance', type=int, default=None, help='Test Instance size.')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer layers.')
    parser.add_argument('--d_ff', type=int, default=2048, help='FeedForward dimension.')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='Dropout probability.')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for loss.')
    parser.add_argument('--csv_root_dir', type=str, default='data_info.csv', help='Root directory of train dataset.')
    parser.add_argument('--save_path', type=str, default='model_check_point', help='Path to save model checkpoints.')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model.')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weight or random inintial weight')
    parser.add_argument('--loss', type=str, default='distance', help='Select loss function method (max, smooth, distance, bce)')

    args = parser.parse_args()
    main(args)
