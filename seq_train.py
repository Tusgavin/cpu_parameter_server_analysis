import torch.nn.functional as F
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import sys

BATCH_SIZE = int(sys.argv[2])
EPOCHS = int(sys.argv[3])

info_per_epoch = []
train_info = {}

def get_preprocessed_data_CIFAR10():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    training_set = torchvision.datasets.CIFAR10(
        './data', train=True, transform=preprocess, download=True)
    validation_set = torchvision.datasets.CIFAR10(
        './data', train=False, transform=preprocess, download=True)

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))
    
    return training_loader, validation_loader

def train_one_epoch(epoch_index, model, training_loader, loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.
    imgs_trained = 0

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print("running_loss: {}".format(type(running_loss))) # DEBUG
        imgs_trained += len(inputs) 
        # print("imgs_trained: {}".format(type(imgs_trained))) # DEBUG
        # break # DEBUG
        
        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            # print("last_loss: {}".format(type(last_loss))) # DEBUG
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss, imgs_trained

def train_epochs(model, training_loader, loss_fn, optimizer, validation_loader):
    best_vloss = 1_000_000.   
    start_train_time = time.time()

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch))
 
        epoch_data = {}
        epoch_data['epoch'] = epoch + 1
    
        time_start_epoch = time.time()
        model.train(True)
        avg_loss, imgs_trained = train_one_epoch(epoch + 1, model, training_loader, loss_fn, optimizer)
        model.train(False)
        time_end_epoch = time.time()
        
        epoch_data['epoch_time'] = time_end_epoch - time_start_epoch
        epoch_data['time_from_start'] = time_end_epoch - start_train_time
        epoch_data['images_per_sec'] = imgs_trained / (time_end_epoch - time_start_epoch)

        running_vloss = 0.0
        total = 0.
        correct = 0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            _, predicted = torch.max(voutputs.data, 1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()
            # print("running_vloss: {}".format(type(running_vloss))) # DEBUG
        vacc = 100 * correct / total

        avg_vloss = running_vloss / (i + 1)
        # print("running_vloss: {}".format(type(avg_vloss))) # DEBUG
        
        epoch_data['vloss'] = avg_vloss
        epoch_data['tloss'] = avg_loss
        epoch_data['vacc'] = vacc
        
        info_per_epoch.append(epoch_data)
        
        print('LOSS train {} valid {} ACC {}'.format(avg_loss, avg_vloss, vacc))
        # break # DEBUG
        
    end_train_time = time.time()
    train_info['time'] = end_train_time - start_train_time
    print("Training took: {} sec".format(end_train_time - start_train_time))
    
def test_model(model, validation_loader):
    model.eval()
    total = 0.
    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            inputs, targets = data

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
    return accuracy

def load_model(type_model):
    if type_model == "alexnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return model, loss_fn, optimizer
    
    elif type_model == "resnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, loss_fn, optimizer
    
    else:
        print("Not a valid model")
        exit()

def main():
    # Load chosen model and its loss function and optimizer
    model, loss_fn, optimizer = load_model(sys.argv[1]) 
    
    # Get CIFAR10 data
    training_loader, validation_loader = get_preprocessed_data_CIFAR10()
    
    # Train model
    train_epochs(model, training_loader, loss_fn, optimizer, validation_loader)
    
    # Test final model accuracy
    acc = test_model(model, validation_loader)
    train_info['acc'] = acc
    
    filename = sys.argv[1] + ".txt"
    with open(filename, 'w') as fp:
        fp.write("%s\n" % train_info)
        for item in info_per_epoch:
            fp.write("%s\n" % item)
    print("Done")

if __name__ == "__main__":
    main()