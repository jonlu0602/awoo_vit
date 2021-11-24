import numpy as np
import torch

from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, use_gpu):

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = model.cuda()
        else:
            self.model = model
    def train(self, epochs, train_loader, valid_loader, save_path):
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            for data, label in tqdm(train_loader):
                if self.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                else:
                    data = data
                    label = label                   

                output = self.model(data)
                loss = self.criterion(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)
            print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")
            if epoch % 1 == 0:
              self.val(valid_loader)
        model.save(self.model.state_dict(), save_path)

    def val(self, valid_loader):
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in tqdm(valid_loader):
                if self.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                else:
                    data = data
                    label = label

                val_output = self.model(data)
                val_loss = self.criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
            print(f"- val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")