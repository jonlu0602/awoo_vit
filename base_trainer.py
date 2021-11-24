import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

def draw_result(curve_list1, curve_list2, file_name):
    plt.plot(range(len(curve_list1)), curve_list1, '-b', label='train')
    plt.plot(range(len(curve_list2)), curve_list2, '-r', label='val')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(file_name.split('.')[0])
    plt.savefig(file_name)
    plt.clf()

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
        max_val_acc = 0
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            count = 0
            for data, label in tqdm(train_loader):
                count += 1
                if count == 10:
                  break
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
                epoch_val_loss, epoch_val_accuracy = self.val(valid_loader)
                if epoch_val_accuracy > max_val_acc:
                    print('model saved !!!')
                    torch.save(self.model.state_dict(), save_path)
                val_loss.append(epoch_val_loss)
                val_acc.append(epoch_val_accuracy)
            train_loss.append(epoch_loss)
            train_acc.append(epoch_accuracy)
        ## draw images
        draw_result(train_loss, val_loss, 'val_curve.png')
        draw_result(train_acc, val_acc, 'acc_curve.png')
        

    def val(self, valid_loader):
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            count = 0
            for data, label in tqdm(valid_loader):
                count += 1
                if count == 10:
                  break
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
        return epoch_val_loss, epoch_val_accuracy