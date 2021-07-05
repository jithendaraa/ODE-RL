import os
import wandb
from EncoderDecoder import EncoderDecoder
from data.mm import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from tqdm import tqdm
import numpy as np
import argparse
from torchvision.utils import make_grid
from torchvision.utils import save_image

os.environ["WANDB_API_KEY"] = "73b7aa2bb830c99a8a3e6228588c6587d037ee96"
os.environ["WANDB_MODE"] = "dryrun"

def train(args):
    '''
    main function to run the training
    '''
    notes = 'Encoder -> 3 CGRU, Decoder -> 3 CGRU, 2Conv'
    run = wandb.init(name='SingleConvGRU_5', project='VideoODE', notes=notes,reinit=True)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(args, device)
        
    model.to(device)
    wandb.watch(model)
    cur_epoch = 0

    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=4,verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    for epoch in range(cur_epoch, args.epochs + 1):
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            model.train()
            pred = model(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optimizer.step()
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = model(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
            original_frames = label[0,:,:,:,:].cpu().numpy()
            new_predictions = pred[0,:,:,:,:].cpu().numpy()    
            original_frames = np.array(original_frames)*255
            new_predictions = np.array(new_predictions)*255
            
            wandb.log({'Original': wandb.Image(make_grid(label[0,:,:,:,:].cpu())), "epoch":epoch})
            wandb.log({'Predicted': wandb.Image(make_grid(pred[0,:,:,:,:].cpu())), "epoch":epoch})
            wandb.log({'Original_gif': wandb.Video(original_frames, format="gif"), "epoch":epoch})  
            wandb.log({'Predicted_gif': wandb.Video(new_predictions, format="gif"), "epoch":epoch})    
                
        torch.cuda.empty_cache()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        wandb.log({'train_loss': train_loss, "epoch":epoch})
        wandb.log({'valid_loss': valid_loss, "epoch":epoch})             

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

    run.finish()     
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='mini-batch size')
    parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
    parser.add_argument('-seq_len',
                        default=10,
                        type=int,
                        help='sum of input frames')
    parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
    parser.add_argument('--n_layers', default=3, type=int, help='# layers')
    args = parser.parse_args()

    random_seed = 1996
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    trainFolder = MovingMNIST(is_train=True,
                            root='data/',
                            n_frames_input=args.seq_len,
                            n_frames_output=args.seq_len,
                            num_objects=[3])
    validFolder = MovingMNIST(is_train=False,
                            root='data/',
                            n_frames_input=args.seq_len,
                            n_frames_output=args.seq_len,
                            num_objects=[3])
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                            batch_size=args.batch_size,
                                            shuffle=False)
    validLoader = torch.utils.data.DataLoader(validFolder,
                                            batch_size=args.batch_size,
                                            shuffle=False)
    train(args)
