# TO RUN DISTRIBUTED:
# Main machine: ray start --head --dashboard-port=8888
# Workers: ray start --address='<main_machine_address>'

# Para criar modulos, devemos adicionar em todas as máquinas
# e atualizar o PATH em cada uma

import sys
import os
import ray
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
import torch.nn as nn

class NN():
   def __init__(self, model, optimizer, loss_fn):
      self.model = model
      self.optimizer = optimizer
      self.loss_fn = loss_fn

   def get_weights(self):
      return {k: v.cpu() for k, v in self.model.state_dict().items()}

   def set_weights(self, weights):
      self.model.load_state_dict(weights)

   def get_gradients(self):
      grads = []
      for p in self.model.parameters():
         grad = None if p.grad is None else p.grad.data.cpu().numpy()
         grads.append(grad)
      return grads

   def set_gradients(self, gradients):
      for g, p in zip(gradients, self.model.parameters()):
         if g is not None:
               p.grad = torch.from_numpy(g)


def get_data_loader(bs):
   preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225]),
   ])

   # Create datasets for training & validation, download if necessary
   with FileLock(os.path.expanduser("~/data.lock")):
      training_set = datasets.CIFAR10(
         './data', train=True, transform=preprocess, download=True)
      validation_set = datasets.CIFAR10(
         './data', train=False, transform=preprocess, download=True)

      training_loader = torch.utils.data.DataLoader(
         training_set, batch_size=bs, shuffle=True, num_workers=2)
      validation_loader = torch.utils.data.DataLoader(
         validation_set, batch_size=bs, shuffle=False, num_workers=2)

   return training_loader, validation_loader


def write_results(data):
   with FileLock(os.path.expanduser("./results.lock")):
      with open("results.txt", "a") as f:
         f.write(data)
         f.close()


def evaluate(model, test_loader):
   print('Validating...')
   model.eval()
   total = 0.
   correct = 0.
   with torch.no_grad():
      for i, data in enumerate(test_loader):
         inputs, targets = data

         outputs = model(inputs)
         _, predicted = torch.max(outputs.data, 1)
         total += targets.size(0)
         correct += (predicted == targets).sum().item()

      accuracy = 100 * correct / total
   return accuracy


@ray.remote
class ParameterServer(object):
   def __init__(self, nn):
      self.nn = nn

   def apply_gradients(self, *gradients):
      # Accumulate gradients
      summed_gradients = [
         np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
      ]
      self.nn.optimizer.zero_grad()
      self.nn.set_gradients(summed_gradients)
      self.nn.optimizer.step()
      # Return new weights
      return self.nn.get_weights()

   def get_weights(self):
      return self.nn.get_weights()


@ray.remote
class DataWorker(object):
   def __init__(self, ps, nn, epochs, train_loader, test_loader, id):
      self.ps = ps
      self.nn = nn
      self.epochs = epochs
      self.train_loader = train_loader
      self.test_loader = test_loader
      self.id = id
      self.info_per_epoch = []
      self.train_info = {}

   def compute_gradients(self, e):
      running_loss = 0.
      last_loss = 0.
      imgs_trained = 0

      # Iterate through training data
      for i, data in enumerate(self.train_loader):
         # Get weights from parameter server
         self.nn.set_weights(ray.get(self.ps.get_weights.remote()))
         inputs, labels = data
         # Fit model
         self.nn.model.zero_grad()
         output = self.nn.model(inputs)
         # Get Loss
         loss = self.nn.loss_fn(output, labels)
         loss.backward()
         # Accumulate loss
         print('Batch: {} Loss {}'.format(i, loss.item()))
         running_loss += loss.item()
         imgs_trained += len(inputs)
         # Send gradiente to parameter server
         self.ps.apply_gradients.remote(self.nn.get_gradients())
      
         if i % 10 == 9:
            batch_info = {}
            last_loss = running_loss / 10  # loss per batch
            print('id {} e{} batch {} loss: {}'.format(self.id, e, i, last_loss))
            batch_info['id'] = self.id
            batch_info['epoch'] = e
            batch_info['batch'] = i
            batch_info['loss'] = last_loss
            with open('output' + str(self.id) + '.txt', 'a') as fp:
               fp.write("%s\n" % batch_info)
            running_loss = 0.
      
      return last_loss, imgs_trained

   def train_epochs(self):
      start_train_time = time.time()

      for epoch in range(self.epochs):
         print('EPOCH {}:'.format(epoch + 1))
 
         epoch_data = {}
         epoch_data['wid'] = self.id
         epoch_data['epoch'] = epoch + 1
      
         time_start_epoch = time.time()
         self.nn.model.train(True)
         avg_loss, imgs_trained = self.compute_gradients(epoch + 1)
         self.nn.model.train(False)
         time_end_epoch = time.time()
         
         epoch_data['epoch_time'] = time_end_epoch - time_start_epoch
         epoch_data['time_from_start'] = time_end_epoch - start_train_time
         epoch_data['images_per_sec'] = imgs_trained / (time_end_epoch - time_start_epoch)

         running_vloss = 0.0
         # for i, vdata in enumerate(self.train_loader):
         #    vinputs, vlabels = vdata
         #    self.nn.set_weights(ray.get(self.ps.get_weights.remote()))
         #    voutputs = self.nn.model(vinputs)
         #    vloss = self.nn.loss_fn(voutputs, vlabels)
         #    running_vloss += vloss.item()

         # avg_vloss = running_vloss / (i + 1)
         # print("running_vloss: {}".format(type(avg_vloss))) # DEBUG
         
         # epoch_data['vloss'] = avg_vloss
         epoch_data['tloss'] = avg_loss
        
         with open('output' + str(self.id) + '.txt', 'a') as fp:
            fp.write("%s\n" % epoch_data)
         
         self.info_per_epoch.append(epoch_data)
         
         print('LOSS train {} valid {}'.format(avg_loss))
         # break # DEBUG
         
      end_train_time = time.time()
      self.train_info['time'] = end_train_time - start_train_time
      print("Training took: {} sec".format(end_train_time - start_train_time))
      return self.info_per_epoch, self.train_info

def load_nn(type_model):
    if type_model == "alexnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return NN(model, optimizer, loss_fn)
    
    elif type_model == "resnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return NN(model, optimizer, loss_fn)
    
    else:
        print("Not a valid model")
        exit()

def main():
   # Hyperparameters
   TYPE_MODEL = sys.argv[1]
   BATCH_SIZE = int(sys.argv[2])
   EPOCHS = int(sys.argv[3])
   NUM_WORKERS = int(sys.argv[4])

   # Load NN, loss func and optimizer
   nn = load_nn(TYPE_MODEL)

   # Init remote actors
   ray.init(ignore_reinit_error=True)
   ps = ParameterServer.remote(nn)
   workers = [
      DataWorker.remote(ps, nn, EPOCHS, get_data_loader(BATCH_SIZE)[0], get_data_loader(BATCH_SIZE)[1], str(i)+str(BATCH_SIZE)+TYPE_MODEL) for i in range(NUM_WORKERS)
   ]

   start_train_time = time.time()

   # Train with each worker
   indo = ray.get([worker.train_epochs.remote() for worker in workers])

   end_train_time = time.time() - start_train_time


   training_loader, test_loader = get_data_loader(BATCH_SIZE)
   current_weights = ps.get_weights.remote()
   nn.set_weights(ray.get(current_weights))
   accuracy = evaluate(nn.model, test_loader)
   print("Final accuracy is {:.1f}.".format(accuracy))
   with open('output.txt', 'w') as fp:
            fp.write("%s\n" % accuracy)

   # Clean up Ray resources and processes before the next example.
   ray.shutdown()
   
   print("----- Took %s seconds to train -----" % end_train_time)


if __name__ == "__main__":
    main()
