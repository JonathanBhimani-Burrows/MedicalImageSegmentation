from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Train_Unet.model.unet import unet
from Train_Unet.experiment.experiment import experiment_class
from Train_Unet.model.unet import enc
import os
import csv



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
    model = GAN_experiment_class(num_classes, batch_size, img, src,lr=lr,device=device, model_path=model_path)
    print("Model Created")
    return model

class GAN_experiment_class(experiment_class):
    '''
    a neural network class which encompasses the training and validation loops and weight initalization
    inherited from the experiment class in the unet
    '''

    def __init__(self, num_classes, batch_size, img, src, lr=0.001, device='cpu', model_path=None):
        # Set up model
        self.lut = [0, 0, 0, 56, 56, 48, 112, 112, 96, 0, 128, 0, 48, 128, 32, 56, 152, 48, 96, 128, 64, 104, 152, 80]
        self.CE = nn.CrossEntropyLoss()
        self.BCE = nn.BCELoss()
        self.L1 = nn.L1Loss()
        self.dummy = img
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.src = src
        self.Gen = unet(self.dummy, num_out=len(num_classes))
        if model_path:
            self.model = self.Gen
            self.load(model_path)
            self.model = self.model.to(self.device)
        else:
            self.Gen.apply(self.init_weights)
            print("Generator Parameters initialized")
            self.Gen = self.Gen.to(self.device)
            self.enc = enc(dummy=self.dummy, disc=True)
            self.calc_output_conv()
            self.Disc = nn.Sequential(self.enc, nn.Conv2d(self.output_dim[1], 1, kernel_size=self.output_dim[2], padding=0, stride=1), nn.Sigmoid())
            self.Disc.apply(self.init_weights)
            print("Discriminator Parameters initialized")
            self.Disc = self.Disc.to(self.device)
            self.model = nn.ModuleList([self.Gen, self.Disc])
            self.optimizer = (torch.optim.Adam(self.model[0].parameters(), lr=lr), torch.optim.Adam(self.model[1].parameters(), lr=lr))
        self.print_summary()
        self.g_train_losses = []
        self.d_train_losses = []
        self.valid_losses = []
        self.tr_dice = []
        self.tr_jacc = []
        self.val_dice = []
        self.val_jacc = []

    def train(self, loaders, lr=1e-3, num_epochs=25):
        '''
        Wrapper function for training on training set + evaluation on validation set.
        '''
        self.make_output_dir()
        scheduler = (StepLR(self.optimizer[0], step_size=25, gamma=0.1),StepLR(self.optimizer[1], step_size=25, gamma=0.1))
        torch.manual_seed(0)
        train_loader = loaders[0]
        valid_loader = loaders[1]
        print("Starting Training")
        for epoch in range(num_epochs):
            tr_metrics, train_loss = self.train_epoch(train_loader, optimizer=self.optimizer, scheduler=scheduler)
            val_metrics, valid_loss, pred, GT = self.valid_epoch(valid_loader, epoch=epoch)
            self.save_state_dicts(epoch)
            self.write_results(epoch)
            # For logging
            self.g_train_losses.append(train_loss[0])
            self.d_train_losses.append(train_loss[1])
            self.valid_losses.append(valid_loss)
            self.tr_dice.append(tr_metrics[0])
            self.tr_jacc.append(tr_metrics[1])
            self.val_dice.append(val_metrics[0])
            self.val_jacc.append(val_metrics[1])
            print('Epoch {}: \t train_dice: {:.2f} \t train_jacc: {:.2f} \t val_dice: {:.2f} \t val_jacc: {:.2f} \t '
                  'generator train_loss: {:.2f} \t discriminator fake train_loss: {:.2f} \t discriminator real train_loss: {:.2f} \t valid_loss: {:.2f}'.format(epoch,
                   tr_metrics[0],tr_metrics[1], val_metrics[0], val_metrics[1], train_loss[0], train_loss[1],train_loss[2], valid_loss))
        self.plot_and_save()


    def train_epoch(self, train_loader, optimizer=None, scheduler=None):
        '''
        Does training for one epoch.
        '''
        self.model.train()
        g_epoch_loss = 0.0
        d_real_epoch_loss = 0.0
        d_gen_epoch_loss = 0.0
        num_batches = 0
        pred = []
        GT = []
        for i, (x, _, y) in enumerate(tqdm(train_loader)):
            # Forward pass
            x = x.to(self.device)
            y = y.to(self.device)
            #infer on generator first
            generator_output = self.model[0](x)
            '''
            Discriminator Network update
            '''
            # create real and fake labels
            label = self.create_label(x.shape)
            batch = self.batchify(x, y, generator_output)
            #fake/generator data
            optimizer[1].zero_grad()
            #fake batch
            discriminator_out_fromgenerator = self.model[1](batch[0].detach())
            dicriminator_loss_fake = self.BCE(discriminator_out_fromgenerator.squeeze(), label[0].squeeze())
            #real batch
            discriminator_out_real = self.model[1](batch[1])
            dicriminator_loss_real = self.BCE(discriminator_out_real.squeeze(), label[1].squeeze())
            #combine losses
            d_loss = (dicriminator_loss_fake + dicriminator_loss_real) / 2
            d_loss.backward()

            '''
            Generator Network Update
            '''
            optimizer[0].zero_grad()
            #as we don't have retain_graph = true, we need to re-infer
            discriminator_out_fromgenerator = self.model[1](batch[0])
            generator_loss = self.BCE(discriminator_out_fromgenerator, label[1])
            # run generator once every n batches, otherwise discriminator loss blows up
            if i % 2 == 0:
                generator_loss.backward()
                optimizer[0].step()
            optimizer[1].step()
            d_real_epoch_loss += dicriminator_loss_real.detach().cpu().numpy()
            d_gen_epoch_loss += dicriminator_loss_fake.detach().cpu().numpy()
            g_epoch_loss += generator_loss.detach().cpu().numpy()
            pred.append(np.argmax(nn.Softmax(dim=1)(generator_output.detach().cpu()).numpy(),axis=1).reshape(-1))
            GT.append(y.detach().cpu().numpy().reshape(-1))
            num_batches += 1
        return self.calc_metrics(pred, GT), (g_epoch_loss/num_batches,d_gen_epoch_loss/num_batches,d_real_epoch_loss/num_batches)


    def valid_epoch(self, valid_loader, epoch=None):
        '''
        Does evaluation on the validation set for one epoch.
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
                loss = self.CE(output, y.long().squeeze(1))
                pred.append(np.argmax(nn.Softmax(dim=1)(output.detach().cpu()).numpy(),axis=1))
                GT.append(y.detach().cpu().numpy())
                num_batches += 1
                epoch_loss += loss.detach().cpu().numpy()
            self.visualize_and_save(pred, GT, epoch)
            return self.calc_metrics(pred, GT), epoch_loss/num_batches, pred, GT


    def create_label(self, disc_shape):
        '''
        a method that creates True and False labels in the shape of the input
        :param disc_shape: shape of the discriminator output
        :return: a tensor of 0's and a tensor of 1's
        '''
        return torch.full((disc_shape[0],1,1,1), 0,dtype=torch.float, device=self.device),\
               torch.full((disc_shape[0],1,1,1), 1,dtype=torch.float, device=self.device)


    def calc_output_conv(self):
        '''
        a method that calculates the output dimension
        :return: 0
        '''
        self.output_dim = self.enc(self.dummy.repeat(1,len(self.num_classes)+1,1,1)).shape



    def batchify(self,x,y,out):
        '''
        a method that creates a batch along the channel axis
        :param x: input image
        :param y: true label
        :param out: prediction from generator
        :return: 1 tensor that is a concat of input image and the prediction, and 1 tensor that is a concat of the input image and the true label
        '''
        return [torch.cat((x,out),dim=1),torch.cat((x,torch.nn.functional.one_hot(y.long(),
                            num_classes=len(self.num_classes)).squeeze(1).permute(0,3,1,2).float()),dim=1)]


    def load(self, model_path):
        '''
        a method that loads model and optimizer state dicts
        :param model_path: path of model state dict
        :param opt_path: path of optimizer state dict
        :return: 0
        '''
        self.model.load_state_dict(torch.load(model_path))
        self.out_path = os.path.join(os.path.split(model_path)[0], "deploy_validation_check")
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

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
            writer.writerow(["Epoch", "Generator Train Loss" , "Discriminator Train Loss","Valid Loss",
                             "Train Dice", " Valid Dice","Train Jaccard","Valid Jaccard"])
            for idx, v in enumerate(self.g_train_losses):
                writer.writerow([e[idx], v, self.d_train_losses[idx] ,self.valid_losses[idx],
                                 self.tr_dice[idx], self.val_dice[idx], self.tr_jacc[idx], self.val_jacc[idx]])


    def save_state_dicts(self, epoch):
        '''
        a method that saves the model and optimizer state dicts
        :param epoch: current epoch
        :return: 0
        '''
        torch.save(self.model[0].state_dict(), os.path.join(self.out_path, "model_state_dict_epoch_" + str(epoch)))


    def plot_and_save(self):
        '''
        a method that plots the data and saves it in the output dir
        :return: 0
        '''
        fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,10))
        ax1.plot(self.g_train_losses, color='red', label='Generator Train Loss', linestyle='dashed')
        ax1.plot(self.d_train_losses, color='red', label='Discriminator Train Loss', linestyle='dotted')
        ax1.plot(self.valid_losses, color='blue', label='Validation Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_xlabel("Loss")
        ax1.set_title("Losses")
        ax1.legend()
        ax2.plot(self.tr_dice, color='red', label='Train Dice Index')
        ax2.plot(self.val_dice, color='blue', label='Validation Dice Index')
        ax2.set_title("Dice Score")
        ax2.set_xlabel("Epochs")
        ax2.set_xlabel("Dice Scores")
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax3.plot(self.tr_jacc, color='red', label='Train Jaccard Score')
        ax3.plot(self.val_jacc, color='blue', label='Validation Jaccard Score')
        ax3.set_title("Jaccard Score")
        ax3.set_xlabel("Epochs")
        ax3.set_xlabel("Jaccard Scores")
        ax3.set_ylim([0, 1])
        ax3.legend()
        plt.savefig(os.path.join(self.out_path,"learning_curves.jpg"))