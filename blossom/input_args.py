import argparse


def get_users_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--input', type=str, default='/', help='path to test image')
    parser.add_argument('--data_directory', type=str, default='flowers/', help='path to folder of images')
    parser.add_argument('--checkpoint', type=bool, default=True, help='Save the model state_dict after training, highly encouraged')
    
    parser.add_argument('--arch',type = str, default='squeezenet', help = 'Type of pre-trained neural network architecture to use, not fully supported')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='path to save the checkpoint')    
    parser.add_argument('--top_k', type=int, default=5, help='Return K most likely class')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to folder of mapping of categories to the real names')
    parser.add_argument('--device', type=str, default='cpu', help='enable gpu training or cpu training, gpu is highly recommended')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='path to folder of images')
    parser.add_argument('--epochs', type=int, default=5, help='number of times to train the model')
    
    parser.add_argument('--train_batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--test_batchsize', type=int, default=16, help='test batch size')

    parser.add_argument('--num_classes', type=int, default=102, help='number of classes in your data')
    parser.add_argument('--trainable', type=bool, default=False, help='Freeze the earlier parameter of the pretrain model when True')
    
    return parser.parse_args()







