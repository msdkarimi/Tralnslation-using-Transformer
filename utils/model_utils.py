import torch
import os
from configs.args import logger
import torch.nn as nn


class ModelUtil:

    def __init__(self, ):
        pass


    def __repr__(self):
        return f'*__model__* : {self.model}\n' \
               f'*__criterion__* : {self.criterion}\n' \
               f'*__optimizer__* : {self.optimizer}\n' \
               f'*__scheduler__* : {self.scheduler}\n'

    def save_checkpoint(self, checkpoint_path, loss, epoch, type='regular'):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': loss,
            'epoch': epoch
        }, checkpoint_path)
        logger.info(f'-----------------epoch {epoch}, saving checkpoint with loss {loss}, type={type}')

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f'-----------------Loading checkpoint with loss {checkpoint["loss"]}, epoch {checkpoint["epoch"]}')

            return checkpoint['loss'], checkpoint['epoch'] + 1

        else:
            logger.info('{')
            return 0, 0

            # raise FileExistsError(f'file {checkpoint_path} does not exists!')

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_model_config(self):
        pass
        # parameters, layers, optimizer, schediler, task