import torch
import os
from configs.args import logger
import torch.nn as nn


class ModelUtil:

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def __repr__(self):
        return f'*__model__* : {self.model}\n' \
               f'*__criterion__* : {self.criterion}\n' \
               f'*__optimizer__* : {self.optimizer}'

    def save_checkpoint(self, checkpoint_path, loss, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'epoch': epoch
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print("vdvdvd")
            logger.info('-----------------Loading checkpoint!')
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            return checkpoint['loss'], checkpoint['epoch']

        else:

            return 0, 0

            # raise FileExistsError(f'file {checkpoint_path} does not exists!')
