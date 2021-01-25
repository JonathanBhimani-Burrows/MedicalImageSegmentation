from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Train_Unet.model.unet import unet
import os
from sklearn.metrics import jaccard_score, f1_score
from PIL import Image
import csv
from operator import itemgetter
import SimpleITK as sitk



def create_model(num_classes, batch_size, lr, img, src, device, model_path=None):
    '''
    a function that creates the model to be trained
    :param num_classes: number of classes in the dataset
    :param batch_size: obvious
    :param lr: learning rate
    :param img: a test image for network sizing
    :param src: source folder location
    :param device: device
    :param model_path: model path, if it exists
    :return: created model
    '''
    model = experiment_class(num_classes, batch_size, img, src ,lr=lr, device=device, model_path=model_path)
    print("Model Created")
    return model


class experiment_class():
    '''
    an experiment class which encompasses the training and validation loops along with several other methods
    '''
    def __init__(self, num_classes, batch_size, img, src,lr=0.0001, device='cpu', model_path=None):
        # Set up model
        self.lut = [0, 0, 0, 56, 56, 48, 112, 112, 96, 0, 128, 0, 48, 128, 32, 56, 152, 48, 96, 128, 64, 104, 152, 80]
        self.loss_fn = nn.CrossEntropyLoss()
        self.dummy = img
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.src = src
        self.model = unet(self.dummy, num_out=len(num_classes))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if model_path:
            self.load(model_path)
            print("Model State Dict Loaded")
        else:
            self.model.apply(self.init_weights)
            print("Parameters initialized")
        self.model = self.model.to(self.device)
        self.print_summary()
        self.train_losses = []
        self.valid_losses = []
        self.tr_dice = []
        self.tr_jacc = []
        self.val_dice = []
        self.val_jacc = []



    def print_summary(self):
        '''
        a method that prints a summary of the model along with the number of parameters
        :return:
        '''
        print(self.model)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of parameters: {}'.format(params))


    def init_weights(self ,m):
        '''
        a method that initializes the weights
        '''
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)



    def train(self, loaders, num_epochs=15):
        '''
        Wrapper method for training on training set + evaluation on validation set.
        '''
        self.make_output_dir()
        scheduler = StepLR(self.optimizer, step_size=12, gamma=0.1)
        torch.manual_seed(0)
        train_loader = loaders[0]
        valid_loader = loaders[1]
        print("Starting Training")
        for epoch in range(num_epochs):
            tr_metrics, train_loss = self.train_epoch(train_loader, optimizer=self.optimizer, scheduler=scheduler)
            val_metrics, valid_loss, pred, GT = self.valid_epoch(valid_loader, epoch=epoch)
            self.visualize_and_save(pred, GT, epoch)
            self.save_state_dicts(epoch)
            self.write_results(epoch)
            scheduler.step()
            print("Current Learning Rate is ",scheduler.get_last_lr())
            # For logging
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.tr_dice.append(tr_metrics[0])
            self.tr_jacc.append(tr_metrics[1])
            self.val_dice.append(val_metrics[0])
            self.val_jacc.append(val_metrics[1])
            print('Epoch {}: \t train_dice: {:.2f} \t train_jacc: {:.2f} \t val_dice: {:.2f} \t val_jacc: {:.2f} \t train_loss: {:.2f} \t'
                  ' valid_loss: {:.2f}'.format(epoch, tr_metrics[0],tr_metrics[1], val_metrics[0], val_metrics[1], train_loss, valid_loss))
        self.plot_and_save()


    def train_epoch(self, train_loader, optimizer=None, scheduler=None):
        '''
        method that performs training for one epoch.
        :param train_loader: obvious
        :param optimizer: obvious
        :param scheduler: obvious
        :return: caluculated metrics, train losses
        '''
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        pred = []
        GT = []
        for i, (x,_,y) in enumerate(tqdm(train_loader)):
            # Forward pass
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            output = self.model(x)
            # Backward pass - loss is sensitive to ordering
            loss = self.loss_fn(output, y.long().squeeze(1))
            loss.backward()
            optimizer.step()
            #store predictions for metric calculation
            pred.append(np.argmax(nn.Softmax(dim=1)(output.detach().cpu()).numpy(),axis=1).reshape(-1))
            GT.append(y.detach().cpu().numpy().reshape(-1))
            epoch_loss += loss.detach().cpu().numpy()
            num_batches += 1
        return self.calc_metrics(pred, GT), epoch_loss/num_batches

    def valid_epoch(self, valid_loader, epoch=None):
        '''
        Does evaluation on the validation set for one epoch.
        :param valid_loader: obvious
        :param epoch: obvious
        :return: calculated metrics, loss, predictions and ground truths
        '''
        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            num_batches = 0
            pred = []
            GT = []
            for i, (x,_, y) in enumerate(tqdm(valid_loader)):
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss_fn(output, y.long().squeeze(1))
                # store predictions for metric calculation
                pred.append(np.argmax(nn.Softmax(dim=1)(output.detach().cpu()).numpy(),axis=1))
                GT.append(y.detach().cpu().numpy())
                num_batches += 1
                epoch_loss += loss.detach().cpu().numpy()
            return self.calc_metrics(pred, GT), epoch_loss/num_batches, pred, GT

    def test_epoch(self, test_loader):
        '''
        Does evaluation on the test set for one epoch.
        :param test_loader: test data loader
        :return: predictions and image names for reconstruction
        '''
        self.model.eval()
        with torch.no_grad():
            pred = []
            names = []
            for i, (x,n,_) in enumerate(tqdm(test_loader)):
                x = x.to(self.device)
                output = self.model(x)
                pred.append(np.argmax(nn.Softmax(dim=1)(output.detach().cpu()).numpy(), axis=1))
                names.append(n)
            return pred, names


    def save_state_dicts(self, epoch):
        '''
        a method that saves the model and optimizer state dicts
        :param epoch: current epoch
        :return: 0
        '''
        torch.save(self.model.state_dict(),os.path.join(self.out_path,"model_state_dict_epoch_"+str(epoch)))


    def load(self,model_path):
        '''
        a method that loads model and optimizer state dicts
        :param model_path: path of model state dict
        :param opt_path: path of optimizer state dict
        :return: 0
        '''
        self.model.load_state_dict(torch.load(model_path))
        self.out_path = os.path.join(os.path.split(model_path)[0],"deploy_validation_check")
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

    def make_output_dir(self):
        '''
        a method that checks if an output directory is already created and if not, creates it
        :return: 0
        '''
        n = np.arange(1000)
        for i in n:
            p = os.path.join(self.src,"V"+str(i))
            if not os.path.exists(p):
                os.makedirs(p)
                self.out_path = p
                break

    def calc_metrics(self, pred, GT):
        '''
        a method that calculates the necessary metrics for this experiment
        :param pred: predicted maps
        :param GT: GT maps
        :return: Dice and Jaccard scores
        '''
        dice = []
        jacc = []
        for idx, prd in enumerate(pred):
            # jaccard score
            jacc.append(jaccard_score(GT[idx].reshape(-1), prd.reshape(-1), average='macro'))
            #Dice score
            dice.append(f1_score(GT[idx].reshape(-1), prd.reshape(-1), average='macro'))
        return (sum(dice)/len(dice), sum(jacc)/len(jacc))


    def deploy(self, loaders):
        '''
        a method that runs a validation epoch to verify parameters, then runs an epoch on the test data
        :param loaders: dataloaders
        :return: 0
        '''
        valid_loader = loaders[1]
        test_loader = loaders[2]
        metrics ,_,_,_ = self.valid_epoch(valid_loader)
        print("Loaded Model Validation metrics are {:.2f} and {:.2f}".format(metrics[0],metrics[1]))
        preds, names = self.test_epoch(test_loader)
        self.save_CT(preds,names)

    def save_CT(self, preds, names):
        '''
        a method that combines the 2D predictions into 3D arrays and saves them as DICOM files
        :param preds: 2D prediction arrays
        :param names: names of the 2D slice
        :return: 0
        '''
        idx = [1,3]
        pred_f = [y for x in preds for y in x]
        # obtain the image names alone and split them
        names_f = [itemgetter(*idx)(os.path.basename(y).split("_")) for x in names for y in x]
        _comb = list(zip(names_f, pred_f))
        #sort the image names such that the image names are in order
        s = sorted(_comb, key=lambda tup: int(tup[0][0]))
        r = []
        temp = []
        #this loop sorts the sublists belonging to each image, such that they can be recombined in the correct order
        for idx,n in enumerate(s):
            if idx == 0:
                temp.append(n)
                c = n[0][0]
            elif n[0][0] == c:
                temp.append(n)
            elif int(n[0][0]) > int(c):
                r.append(sorted(temp, key=lambda tup: int(tup[0][1].split(".")[0])))
                c = n[0][0]
                temp = []
                temp.append(n)
            if len(s) == idx + 1:
                r.append(sorted(temp, key=lambda tup: int(tup[0][1].split(".")[0])))
        #combine all the 2D arrays into 3D ones, and then convert into DICOM
        for slices in r:
            name, im = list(zip(*slices))
            header = np.load(os.path.join(self.src,'ct_test\image_header_' + name[0][0] + '.npy'), allow_pickle=True)[()]
            ims = sitk.GetImageFromArray(np.stack(im))
            for k,v in header.items():
                try:
                    if k == 'dim[2]' or k == 'dim[3]':
                        ims.SetMetaData(k, '128')
                    else:
                        ims.SetMetaData(k, v)
                except:
                    pass
            sitk.WriteImage(ims,os.path.join(self.src,'ct_test',"ct_test_prediction_"+name[0][0]+".nii"))



    def visualize_and_save(self,pred, GT, epoch):
        '''
        a method that saves files for visual inspection
        :param pred: predicted maps
        :param GT: GT maps
        :param epoch: current epoch
        :return: 0
        '''
        # visualize some of the images to verify optimization
        t = np.arange(10) if len(pred) > 10 else np.arange(len(pred))
        img_out_path = os.path.join(self.out_path,"epoch"+str(epoch))
        os.mkdir(img_out_path)
        for i in t:
            im_p = Image.fromarray(pred[i][0].astype('uint8'))
            im_p.putpalette(self.lut)
            im_p.save(os.path.join(img_out_path,"pred_"+str(i)+".png"))
            im_gt = Image.fromarray(GT[i][0].squeeze().astype('uint8'))
            im_gt.putpalette(self.lut)
            im_gt.save(os.path.join(img_out_path,"GT_"+str(i)+".png"))

    def write_results(self, epoch):
        '''
        a method that writes the performances metrics into a CSV
        :param epoch: current epoch
        :return:0
        '''
        e = np.arange(epoch)
        result_path = os.path.join(self.out_path, "results.csv")
        with open(result_path, mode='w', newline="") as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Train Dice", " Valid Dice","Train Jaccard","Valid Jaccard"])
            for idx, v in enumerate(self.train_losses):
                writer.writerow([e[idx], v, self.valid_losses[idx], self.tr_dice[idx], self.val_dice[idx], self.tr_jacc[idx], self.val_jacc[idx]])

    def plot_and_save(self):
        '''
        a method that plots and saves the learning curves
        :return: 0
        '''
        fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,10))
        ax1.plot(self.train_losses, color='red', label='Train Loss')
        ax1.plot(self.valid_losses, color='blue', label='Validation Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_xlabel("Loss")
        ax1.set_title("Losses")
        ax1.legend()
        ax2.plot(self.tr_dice, color='red', label='Train Jaccard Index')
        ax2.plot(self.val_dice, color='blue', label='Validation Jaccard Index')
        ax2.set_title("Jaccard Indices")
        ax2.set_xlabel("Epochs")
        ax2.set_xlabel("Jaccard Index")
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax3.plot(self.tr_jacc, color='red', label='Train Dice Score')
        ax3.plot(self.val_jacc, color='blue', label='Train Dice Score')
        ax3.set_title("Dice Score")
        ax3.set_xlabel("Epochs")
        ax3.set_xlabel("Dice Scores")
        ax3.set_ylim([0, 1])
        ax3.legend()
        plt.savefig(os.path.join(self.out_path,"learning_curves.jpg"))
