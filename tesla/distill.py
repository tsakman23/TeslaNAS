import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import augment, get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, DiffAugmentList, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time

import json  # Add this for logging accuracy results

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def main(args):
    # Validate arguments
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    # Compute total experts if both max_experts and max_files are provided
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files
    
    # Start time
    start_time = time.time()

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    # Convert string argument to boolean
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define evaluation intervals
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    
    # Load dataset and related parameters
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]
    args.im_size = im_size

    # Initialize a dictionary to store accuracy results
    accs_all_exps = {key: [] for key in model_eval_pool}

    # List to store synthetic data for potential saving
    data_save = []

    # Configure differentiable Siamese augmentation (DSA) parameters
    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    zca_trans = args.zca_trans if args.zca else None


    # Initialize wandb for logging
    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    # Create an empty object to store wandb configuration
    args = type('', (), {})()

    # Set attributes from wandb configuration
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    # Restore DSA and ZCA parameters
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    # Set batch size for synthetic data if not defined
    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    # Check if multiple GPUs are available
    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    # Prepare dataset index mapping

    indices_class = [[] for c in range(num_classes)]
    # Build label to index map
    print("---------------Build label to index map--------------")
    """
    For machines with limited RAM, it's impossible to load all ImageNet or even TinyImageNet into memory.
    Even if it's possible, it will take too long to process.
    Therefore we pregenerate an indices to image map and use this map to quickly random samples from ImageNet or TinyImageNet dataset.
    """
    if args.dataset == 'ImageNet':
        indices_class = np.load('indices/imagenet_indices_class.npy', allow_pickle=True)
    elif args.dataset == 'Tiny':
        indices_class = np.load('indices/tiny_indices_class.npy', allow_pickle=True)
    else:
        for i, data in tqdm(enumerate(dst_train)):
            indices_class[data[1]].append(i)

    # for c in range(num_classes):
    #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

    def get_images(c, n):
        """
        Retrieve n random images from class c
        """
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        subset = Subset(dst_train, idx_shuffle)
        data_loader = DataLoader(subset, batch_size=n)
        # only read the first batch which has n(IPC) number of images.
        for data in data_loader:
            return data[0].to("cpu")


    ''' initialize the synthetic data with random values or real images '''
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        print('Initializing synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('Initializing synthetic data from random noise')

    print()

    ''' training '''
    # Move syntehtic data to device and make it trainable
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    # Print synthetic dataset statistics
    print(f"Number of synthetic images: {image_syn.shape[0]}")
    print(f"Image dimensions: {image_syn.shape[1:]}\n")

    # Define optimizers for synthetic images and learning rate
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5) # Optimize synthetic data with SGD optimizer with momentum
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()
    optimizer_lr.zero_grad()

    # Define loss criterion
    criterion = nn.CrossEntropyLoss().to(args.device)

    print('-'*50)
    print('%s TRAINING BEGINS'%get_time())
    print()

    # Define expert directory path
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    # Check if precomputed expert trajectories are used
    if not args.random_trajectory:
        if args.load_all: 
            buffer = [] # Load all expert trajectories
            n = 0 # Counter for expert trajectory (buffer) files
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                # Load expert trajectory and append to buffer
                buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

        else:
            expert_files = [] # List to store paths of expert trajectory (buffer) files
            n = 0 # Counter for expert trajectory (buffer) files
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            file_idx = 0 # Track current file index
            expert_idx = 0 # Track current expert within a file
            random.shuffle(expert_files) # Shuffle expert files for random selection
            if args.max_files is not None:
                expert_files = expert_files[:args.max_files] # Limit number of expert files to read
            print("loading file {}".format(expert_files[file_idx]))
            buffer = torch.load(expert_files[file_idx]) # Load first expert file
            if args.max_experts is not None:
                buffer = buffer[:args.max_experts] # Limit number of experts to read
            random.shuffle(buffer) # Shuffle expert trajectories for random selection

    # Initialize best accuracy tracking for different models
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    # Main training loop
    for it in range(0, args.Iteration+1):
        save_this_it = False # Flag to determine if current iteration results should be saved
        wandb.log({"Progress": it}, step=it) # Log current iteration to wandb


        # Periodic evaluation of the synthetic dataset
        if it in eval_it_pool and args.eval_it > 0:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('\nDSA augmentation strategy: ', args.dsa_strategy)
                    print('\nDSA augmentation parameters: ', args.dsa_param.__dict__)
                else:
                    print('\nDC augmentation parameters: ', args.dc_aug_param)

                accs_test = [] # List to store test accuracy results
                accs_train = [] # List to store train accuracy results
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    eval_labs = label_syn # Use synthetic labels for evaluation
                    with torch.no_grad():
                        image_save = image_syn # Copy synthetic images for evaluation
                    # Deepcopy to avoid any unaware modification
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item() # Set synthetic learning rate

                    # Evaluate synthetic dataset using current model
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test) # Store test accuracy
                    accs_train.append(acc_train) # Store train accuracy
                accs_test = np.array(accs_test) 
                accs_train = np.array(accs_train) 
                acc_test_mean = np.mean(accs_test) # Compute mean test accuracy
                acc_test_std = np.std(accs_test) # Compute mean train accuracy

                # Store accuracy results in accs_all_exps
                accs_all_exps[model_eval].append((it, acc_test_mean, acc_test_std))

                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean # Update best accuracy
                    best_std[model_eval] = acc_test_std # Update best standard deviation
                    save_this_it = True # Mark this iteration for saving
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                # Log evaluation results to wandb
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

        # Periodically save synthetic dataset
        if it in eval_it_pool and (save_this_it or it % 1000 == 0) and args.eval_it > 0:
            with torch.no_grad():
                image_save = image_syn.cuda() # Move synthetic images to GPU

                # Save synthetic data to data_save
                data_save.append((it, image_save.cpu(), label_syn.cpu()))

                # Define save directory based on dataset and wandb run name
                save_dir = os.path.join(".", "logged_files", args.dataset, 'offline' if wandb.run.name is None else wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir) # Create save directory if it does not exist

                # Save synthetic images and labels
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)
        
        # Log synthetic learning rate to wandb
        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        # Initialize student network for distillation
        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        student_net = ReparamModule(student_net) # Wrap student network with ReparamModule

        # If using multiple GPUs, wrap student network with DataParallel module for parallel processing
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train() # Set student network to training mode

        # Load an expert trajectory if using precomputed ones
        if not args.random_trajectory:
            if args.load_all:
                expert_trajectory = buffer[np.random.randint(0, len(buffer))]
            else:
                expert_trajectory = buffer[expert_idx]
                expert_idx += 1
                if expert_idx == len(buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del buffer
                        buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        buffer = buffer[:args.max_experts]
                    random.shuffle(buffer)

        # Randomly select a starting epoch for the expert trajectory within the allowed range
        start_epoch = np.random.randint(0, args.max_start_epoch)

        # Retrieve starting and target parameters for distillation from expert trajectory
        if not args.random_trajectory:
            starting_params = expert_trajectory[start_epoch]
            target_params = expert_trajectory[start_epoch+args.expert_epochs]
        else:
            starting_params = [p for p in student_net.parameters()]
            target_params = [p for p in student_net.parameters()]

        # Flatten starting and target parameters for distillation into a single tensor
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        # Compute the parameter distance for optimization reference
        param_dist = torch.tensor(0.0).to(args.device)
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        # Produce soft labels for soft label assignment. This is only used if teacher_label is set to True
        if args.teacher_label:
            label_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
            label_net = ReparamModule(label_net)
            label_net.eval() # Set to evaluation mode

            # use the target param as the model param to get soft labels.
            label_params = copy.deepcopy(target_params.detach()).requires_grad_(False)

            batch_labels = []
            SOFT_INIT_BATCH_SIZE = 50
            if image_syn.shape[0] > SOFT_INIT_BATCH_SIZE and args.dataset == 'ImageNet':
                for indices in torch.split(torch.tensor([i for i in range(0, image_syn.shape[0])], dtype=torch.long), SOFT_INIT_BATCH_SIZE):
                    batch_labels.append(label_net(image_syn[indices].detach().to(args.device), flat_param=label_params))
            else:
                label_syn = label_net(image_syn.detach().to(args.device), flat_param=label_params)
            label_syn = torch.cat(batch_labels, dim=0)
            label_syn = torch.nn.functional.softmax(label_syn)
            del label_net, label_params
            for _ in batch_labels:
                del _

        # Begin training on synthetic images
        syn_images = image_syn
        y_hat = label_syn.to(args.device)

        # Initialize gradient storage for synthetic images
        syn_image_gradients = torch.zeros(syn_images.shape).to(args.device)

        # Lists to store intermediate training data
        x_list = [] # Store augmented images
        original_x_list = [] # Store original synthetic images
        y_list = [] # Store labels for optimization
        indices_chunks = [] # Store indices for synthetic image batches
        gradient_sum = torch.zeros(student_params[-1].shape).to(args.device) # Accumulate gradients for synthetic images over steps
        indices_chunks_copy = [] # Track processed index batches

        # Perform multiple steps of distillation on synthetic images
        for _ in range(args.syn_steps):

            # If indices_chunks is empty, create a new shuffled index list
            if not indices_chunks:
                indices = torch.randperm(len(syn_images)) # Shuffle indices for synthetic images
                indices_chunks = list(torch.split(indices, args.batch_syn)) # Split indices into batches

            these_indices = indices_chunks.pop() # Retrieve a batch of indices
            indices_chunks_copy.append(these_indices) # Store processed indices

            x = syn_images[these_indices] # Retrieve synthetic images for the batch
            this_y = y_hat[these_indices] # Retrieve labels for the batch
            original_x_list.append(x) # Store original synthetic images before augmentation

            # Perform data augmentation on synthetic images if DSA is enabled
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            x_list.append(x.clone()) # Store augmented synthetic images
            y_list.append(this_y.clone()) # Store labels for the batch

            forward_params = student_params[-1] # Retrieve current student parameters
            x = student_net(x, flat_param=forward_params) # Forward pass on student network
            ce_loss = criterion(x, this_y) # Compute loss (cross-entropy)

            # Compute gradient of loss w.r.t. student parameters
            grad = torch.autograd.grad(ce_loss, forward_params, create_graph=True, retain_graph=True)[0]

            # Detach and clone gradient to prevent modification during backpropagation
            detached_grad = grad.detach().clone()

            # Update student model parameters using synthetic learning rate
            student_params.append(student_params[-1] - syn_lr.item() * detached_grad)

            # Accumulate gradients for later use
            gradient_sum += detached_grad

            # Delete gradient tensor to free up memory
            del grad

        # --------Compute the gradients regarding synthetic input image and learning rate---------
        # compute gradients invoving 2 gradients
        for i in range(args.syn_steps):
            # compute gradients for w_i
            w_i = student_params[i]
            output_i = student_net(x_list[i], flat_param = w_i)
            if args.batch_syn:
                ce_loss_i = criterion(output_i, y_list[i])
            else:
                ce_loss_i = criterion(output_i, y_hat)

            grad_i = torch.autograd.grad(ce_loss_i, w_i, create_graph=True, retain_graph=True)[0]
            single_term = syn_lr.item() * (target_params - starting_params)
            square_term = (syn_lr.item() ** 2) * gradient_sum
            gradients = 2  * torch.autograd.grad( (single_term + square_term) @ grad_i / param_dist, original_x_list[i])
            with torch.no_grad():
                syn_image_gradients[indices_chunks_copy[i]] += gradients[0]
        # ---------end of computing input image gradients and learning rates--------------

        syn_images.grad = syn_image_gradients

        grand_loss = starting_params - syn_lr * gradient_sum - target_params
        grand_loss = grand_loss.dot(grand_loss) / param_dist

        lr_grad = torch.autograd.grad(grand_loss, syn_lr)[0]
        syn_lr.grad = lr_grad

        optimizer_img.step()
        optimizer_lr.step()


        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _
        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

# After training, log accuracy results to a file
    accuracy_log_path = os.path.join(".", "logged_files", args.dataset, "accuracy_results.json")
    with open(accuracy_log_path, "w") as f:
        json.dump(accs_all_exps, f, indent=4)

    print(f"Accuracy results saved to {accuracy_log_path}")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    # Log total time taken to wandb
    wandb.log({"Total_Time": end_time - start_time})

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Specifies the dataset to use (default: \'CIFAR10\').')

    parser.add_argument('--model', type=str, default='ConvNet', help='Defines the neural network model architecture (default: \'ConvNet\').')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=3, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=2000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.011, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset directory (default: \'./data\')')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='Path to store buffered data (default: \'./buffers\').')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--random_trajectory', action='store_true', default=False, help="using random trajectory instead of pretrained")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', default=True, help='this will save images for 50ipc')

    parser.add_argument('--teacher_label', action='store_true', default=False, help='whether to use label from the expert model to guide the distillation process.')

    args = parser.parse_args()

    main(args)


