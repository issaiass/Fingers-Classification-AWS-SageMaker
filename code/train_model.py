# Import needed libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np

import argparse

# for sagemaker debug
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

# for load images and payload
from PIL import Image
from io import BytesIO
import json

# cuda or cpu device and hook for model debugging and profiling
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hook = get_hook(create_if_not_exists=True)

# load the model on the submited directory
def model_fn(model_dir):
    print('START MODEL_FN')
    model = Net(num_classes=12)
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    print('END MODEL_FN')
    return model 

# call the inputs and return a PIL image 
def input_fn(request_body, request_content_type):
    print('START INPUT_FN')
    print(request_content_type)
    print(type(request_body))
    print(request_body)
    if request_content_type == 'application/x-image':
        png_start = request_body.find(b'\x89PNG\r\n\x1a\n')
        request_body = request_body[png_start:]
        print('END INPUT_FN')
        return Image.open(BytesIO(request_body))
    else:
        print('can only process images')
        pass

# Predict over the returned input object of the input_fn and the model_fn
def predict_fn(input_object, model):
    print('START PREDICT_FN')
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),        
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_object).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor.to(device))
    print('STOP PREDICT_FN')
    return prediction

# return the results arranged over the predict_fn output
#def output_fn(predictions, content_type, context='application/json'):
def output_fn(predictions, accept):
    print('START OUTPUT_FN')
    classes = np.array(['0L', '0R', '1L', '1R', '2L', '2R', '3L', '3R', '4L', '4R', '5L', '5R'])
    predictions = predictions.cpu().numpy()
    classes_sorted = classes[np.flip(predictions.argsort())]
    predictions.sort()
    predictions_proba = np.flip(predictions)
    json_res = dict(classes=classes_sorted.tolist(), predictions=predictions_proba.tolist())
    if accept == 'application/json':
        print('END OUTPUT_FN')
        json_res = json.dumps(json_res)
    else:
        raise ValueError('Accept header must be application/json')
    return json_res


# testing phase
def test(model, test_loader, criterion, device):
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"test accuracy: {100*total_acc}, test Loss: {total_loss}")


# training phase
def train(model, epochs, train_loader, validation_loader, criterion, optimizer, device):
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                hook.set_mode(modes.TRAIN)                                
                model.train()
            else:
                hook.set_mode(modes.EVAL)
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    hook.set_mode(modes.TRAIN)                       
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 20  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] {} loss: {:.2f} accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            phase,
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                hook.set_mode(modes.EVAL)
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
        if loss_counter==1:
            break
    return model


# Load the model
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) # pytorch > 2.0
        self.model = models.vgg16(pretrained=True)        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 64),
            nn.Dropout(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x


# load the dataset
def create_dataset_loaders(data, batch_size):
    training_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # we will transform grayscale images to 3 channel data because VGG16 is for RGB images
        transforms.Resize(128), #transforms.RandomResizedCrop(128), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),        
        transforms.Resize(128),
        transforms.ToTensor(),
    ])

    testing_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),        
        transforms.Resize(128),
        transforms.ToTensor(),
    ])

    train_data = ImageFolder(root=os.path.join(data, 'train'), transform=training_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_data = ImageFolder(root=os.path.join(data, 'valid'), transform=valid_transform)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    test_data = ImageFolder(root=os.path.join(data, 'test'), transform=testing_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True) 

    return train_loader, valid_loader, test_loader

# main function
def main(args):
    train_loader, valid_loader, test_loader = create_dataset_loaders(args.training, batch_size=args.batch_size)    
    net = Net(num_classes=12)
    net = net.to(device)
    hook.register_module(net)
    criterion = nn.CrossEntropyLoss()
    hook.register_loss(criterion)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate)
    trained_model = train(net, args.epochs, train_loader, valid_loader, criterion, optimizer, device)
    test(model=trained_model, test_loader=test_loader, criterion=criterion, device=device)
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(trained_model.state_dict(), f)

# we will enter here if we train the model
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=0.01, metavar="LR", help="input batch size for testing (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=64, metavar="BS", help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)")

    # Container environment
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    args=parser.parse_args()
    
    main(args) # call the main program with the parsed arguments