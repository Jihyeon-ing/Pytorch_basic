import torch.nn as nn
import os
import glob
import torch
import numpy as np
from model import *
from tqdm import tqdm

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        self.is_train = self.args.is_train

        ## parameters for training ##
        self.batch_size = self.args.batch_size
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            device = 'cuda'
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device
      
    def train(self.args):
        config = args.__dict__
        self.model = MLP(self.batch_size, self.args.in_n, self.args.hidden_n, self.args.out_n)
        self.model.to(self.device)

        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()
  
            n_step = len(self.train_set) // self.batch_size
            idx = list(range(self.train_set))
            random.shuffle(idx)

            for step in tqdm(range(n_step)):
                self.optimizer.zero_grad()
                batch_idx = idx[self.batch_size*step:self.batch_size*(step+1)]
                batch_x, batch_y = load_data(idx=batch_idx, mode='train', **config)
                batch_x = batch_x.to(self.device) 
                batch_y = batch_y.to(batch_x.device)
                pred_y = self.model(batch_x)    # output of the model

                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.mean().item())
                
                loss.backward()
                torch.cuda.synchronize()
                self.optimizer.step()
            
            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                self.model.eval()
                with torch.no_grad()
                    vali_loss = self.vali(args, epoch)
                    if epoch % (args.log_step * 1) == 0:
                          self._save(name=str(epoch))
                print(f"Epoch: {epoch+1}  |  Train Loss: {train_loss:.4f}  Validation Loss: {vali_loss:.4f}")

        best_model_path = f'{self.model_path}/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, args, epoch):
        config = args.__dict__
        pred_list, true_list, total_loss = [], [], []

        n_step = len(self.vali_set) // self.args.val_batch_size
        idx = list(range(self.vali_set))
        random.shuffle(idx)

        for step in tqdm(range(n_step)):
            batch_idx = idx[self.args.val_batch_size*step:self.args.val_batch_size*(step+1)]
            batch_x, batch_y = load_data(idx=batch_idx, mode='val', **config)
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(batch_x.device)
            pred_y = self.model(batch_x)

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), 
                     [pred_y, batch_y], [pred_list, true_list]))

            loss = self.criterion(pred_y, batch_y)
            total_loss.appeend(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(pred_list, axis=0)
        trues = np.concatenate(true_list, axis=0)

        return total_loss

    def test(self, args):
        config = args.__dict__
        epoch = self.args.test_epoch
        model = self.model
        
        ckpt = torch.load(os.path.join(self.checkpoints_path, f'{epoch}.pth'))
        model.load_statd_dict(ckpt)
        model.to(self.device)
        model.eval()

        savedir = os.path.join('results', f'epoch_{epoch}')     # directory for saving results of the model
        os.makedirs(savedir, exist_ok=True)
            
        input_list, true_list, pred_list = [], [], []
        idx = list(range(len(self.test_set)))
        for i in tqdm(idx):
            batch_x, batch_y = laod_data(idx=[i], mode='test', **config)
            pred_y = self.model(batch_x.to(self.device))

            inputs = batch_x.cpu().detach().numpy()
            targets = batch_y.cpu().detach().numpy()
            preds = pred_y.cpu().detach().numpy()
            np.savez_compressed(os.path.join(savedir, f'{str(i).zfill(4)}'), inp=inputs, tar=targets, pred=preds)
        
        
  
