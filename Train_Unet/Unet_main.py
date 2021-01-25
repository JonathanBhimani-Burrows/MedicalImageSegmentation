import numpy as np
import torch
from data_prep.data_prep import load_and_save
from data_prep.make_dataset import dataset_processing
from experiment.experiment import create_model
import argparse


def main():
    #parse args
    src = args.data_path
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    img_size = args.img_size
    lr = args.initial_learning_rate
    paths = ["ct_train","ct_test"]
    norm = 'per_patient'
    #choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is", device)
    #seed
    np.random.seed(42)
    torch.manual_seed(42)
    #loads and saves images
    names, num_classes = load_and_save(src, paths, norm=norm, img_size=img_size)
    #creates datasets and dataloaders
    dls = dataset_processing((src, paths), names, batch_size)
    #deployment
    if args.model_path:
        model = create_model(num_classes, batch_size, lr, iter(dls[0]).next()[0][0].unsqueeze(0), src, model_path=args.model_path, device=device)
        model.deploy(dls)
    #training
    else:
        model = create_model(num_classes, batch_size, lr,iter(dls[0]).next()[0][0].unsqueeze(0), src, device=device)
        model.train(dls, num_epochs=num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("initial_learning_rate", type=float)
    parser.add_argument("--img_size", required=False, type=int, default=512)
    parser.add_argument("--model_path", required=False, default=None)
    args = parser.parse_args()
    main()



