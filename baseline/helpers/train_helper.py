import random
import numpy as np
import torch

from metrics.average_meter import AverageMeter
from metrics.average_ensemble_meter import AverageEnsembleMeter

from models.shapes_trainer import ShapesTrainer

class TrainHelper():
    def __init__(self, device):
        self.device = device

    def train_one_batch(
        self,
        model: ShapesTrainer,
        batch,
        optimizer,
        meta_data,
        device,
        inference_step,
        multi_task):
        """
        Train for single batch
        """
        model.train()

        optimizer.zero_grad()

        target, distractors, indices, md5 = batch

        if inference_step or multi_task:
            md = torch.tensor(meta_data[indices[:,0], :], device=device, dtype=torch.int64)
            # print(meta_data.shape)
            md = md5
            # stop
        else:
            md = None
            
        loss, losses, accuracies, _ = model.forward(target, distractors, md)

        loss.backward()
        optimizer.step()

        return losses, accuracies

    def evaluate(self, model, dataloader, valid_meta_data, device, inference_step, multi_task, step3, property_one, property_two):
        
        if multi_task:
            loss_meter = [AverageEnsembleMeter(5), AverageMeter()]
            acc_meter = [AverageEnsembleMeter(5), AverageMeter()]
        elif inference_step or step3:
            loss_meter = AverageEnsembleMeter(5)
            acc_meter = AverageEnsembleMeter(5)
        else:
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

        messages = []

        model.eval()
        for batch in dataloader:
            # note step3 takes lkey, but for zeroshot this is vmd5, thus step3 and zeroshot can't be combined
            # target, distractors, indices, lkey = batch
            target, distractors, indices, vmd5 = batch
            # print(target.shape, len(distractors))
            # print(property_one, property_two)
            # print(valid_meta_data[indices[:, :], :])
            # one_begin = round(property_one/3)*3
            # one_end = round(property_one/3)*3+3
            # two_begin = round(property_two/3)*3
            # two_end = round(property_two/3)*3+3
            # print(valid_meta_data.shape)
            # print(indices)
            # stop
            # print(valid_meta_data[indices[:, :], :])
            # print(torch.tensor(valid_meta_data[:,one_begin:one_end]).shape,valid_meta_data[indices[:,0],one_begin:one_end].shape)
            # stop

            if inference_step or multi_task:
                vmd = torch.tensor(valid_meta_data[indices[:, 0], :], device=device, dtype=torch.int64)
                # I set this to vmd5 for now, must only be done for zeroshot
                vmd = vmd5
            else:
                vmd = None


            _, loss2, acc, msg = model.forward(target, distractors, vmd)

            # print('helper-loss',loss2)

            if multi_task:
                loss_meter[0].update(loss2[0])
                loss_meter[1].update(loss2[1])

                acc_meter[0].update(acc[0])
                acc_meter[1].update(acc[1])
            elif step3:
                # Note that for the RANDOM step3 dict, lkey is just a random integer
                # Thus its prediction of the classes are average accuracies over smaller sets
                lkey = torch.tensor(list(map(int, lkey)))
                lkey_stack = torch.stack([lkey == 0, lkey == 1, lkey == 2, lkey == 3, lkey == 4])

                acc = (torch.sum(lkey_stack.cpu().float() * acc.cpu().float(), dim=1)/torch.sum(lkey_stack.cpu().float(),dim=1)).numpy()
                loss2 = (torch.sum(lkey_stack.cpu().float() * loss2.cpu().float(), dim=1)/torch.sum(lkey_stack.cpu().float(),dim=1)).detach().numpy()

                acc[np.isnan(acc)] = 0
                loss2[np.isnan(loss2)] = 0

                loss_meter.update(loss2)
                acc_meter.update(acc)
                # print(acc_meter.averages)

            else:
                loss_meter.update(loss2)
                acc_meter.update(acc)

            messages.append(msg)

        return (
            loss_meter,
            acc_meter,
            torch.cat(messages, 0)
        )

    def get_filename_from_baseline_params(self, params):
        """
        Generates a filename from baseline params (see baseline.py)
        """
        if params.name:
            return params.name

        name = params.dataset_type
        name += "_e_{}".format(params.embedding_size)
        name += "_h_{}".format(params.hidden_size)
        name += "_lr_{}".format(params.lr)
        name += "_max_len_{}".format(params.max_length)
        name += "_k_{}".format(params.k)
        name += "_vocab_{}".format(params.vocab_size)
        name += "_seed_{}".format(params.seed)
        name += "_btch_size_{}".format(params.batch_size)
        if params.single_model:
            name += "_single_model"
        if params.greedy:
            name += "_greedy"
        if params.debugging:
            name += "_debug"
        if params.sender_path or params.receiver_path:
            name += "_loaded_from_path"
        if params.inference_step:
            name += "_inference"
        if params.step3:
            name += "_step3"
        if params.multi_task:
            name += "_multi"
            if params.multi_task_lambda:
                name += f'_lambda_{params.multi_task_lambda}'
                
        if params.disabled_properties:
            name += "_disabled"
            for image_property in params.disabled_properties:
                name+= f'_{int(image_property)}'
            
        return name

    def seed_torch(self, seed=42):
        """
        Seed random, numpy and torch with same seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)