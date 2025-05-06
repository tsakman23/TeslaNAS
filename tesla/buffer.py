import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import time

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    # Convert string argument for DSA (Differentiable Siamese Augmentation) to boolean
    args.dsa = True if args.dsa == 'True' else False
    # Set the device to GPU if available, otherwise use CPU
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', args.device)
    # Initialize DSA parameters
    args.dsa_param = ParamDiffAug()

    # Load dataset and related parameters
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)

    # Print hyperparameters for debugging and reproducibility
    print('Hyper-parameters: \n', args.__dict__)

    # Define the directory to save replay buffers
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it doesn't exist

    # Define the loss function based on the specified loss type
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss().to(args.device)  # Cross-entropy loss for classification tasks
    else:
        criterion = nn.MSELoss().to(args.device)  # Mean squared error loss for regression tasks

    trajectories = []  # List to store the trajectories (weights of the model at different epochs)

    # Create a DataLoader for the training dataset
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    # Set data augmentation parameters for whole-dataset training
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # Augmentation strategy
    print('DC augmentation parameters: \n', args.dc_aug_param)

    # Loop to train multiple expert models
    for it in range(0, args.num_experts):
        ''' Train synthetic data '''
        # Initialize a new teacher network (expert model)
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        teacher_net.train()  # Set the model to training mode

        # Initialize the optimizer for the teacher network
        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
        teacher_optim.zero_grad()

        timestamps = []  # List to store the weights of the teacher model at different epochs

        # Save the initial weights of the teacher model
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        # Define a learning rate schedule (reduce learning rate halfway through training)
        lr_schedule = [args.train_epochs // 2 + 1]

        # Train the teacher network for the specified number of epochs
        for e in range(args.train_epochs):
            # Perform one epoch of training
            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                          criterion=criterion, args=args, aug=True)

            # Print training progress
            if e == args.train_epochs - 1:  # Evaluate the model at the end of training
                test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                            criterion=criterion, args=args, aug=False)
                print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it + 1, e + 1, train_acc, test_acc))
            else:
                print("Itr: {}\tEpoch: {}\tTrain Acc: {}".format(it + 1, e + 1, train_acc))

            # Save the weights of the teacher model after each epoch
            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            # Adjust the learning rate if the current epoch is in the learning rate schedule
            if e in lr_schedule and args.decay:
                lr *= 0.1  # Reduce learning rate by a factor of 10
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()

        # Append the trajectory (weights at all epochs) of the current teacher model to the list
        trajectories.append(timestamps)

        # Save the trajectories to disk at regular intervals
        if len(trajectories) == args.save_interval:
            n = 0
            # Find the next available file name
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            # Save the trajectories to a file
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []  # Reset the trajectories list


if __name__ == '__main__':

    start_time = time.time()
    print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--zca', action='store_true', default=False)
    parser.add_argument('--decay', action='store_true', help='whether to decay learning rate')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--loss_type', type=str, default='ce', help='loss type, ce or mse')
    parser.add_argument('--save_interval', type=int, default=1)

    args = parser.parse_args()
    main(args)

    end_time = time.time()
    print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))

    print("Total time: ", end_time - start_time)
