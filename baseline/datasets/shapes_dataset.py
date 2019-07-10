import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler
import torchvision.transforms
from PIL import Image


class ShapesDataset:
    def __init__(
        self, features, mean=None, std=None, metadata=False, raw=False, dataset=None, step3_distractors = None, validation_set = False, valid_meta_data = None, distractor_features = None, distractor_meta_data = None, property_one = None, property_two = None
    ):
        self.metadata = metadata
        self.raw = raw
        self.features = features

        self.obverter_setup = False
        self.dataset = dataset
        self.step3_distractors = step3_distractors
        self.validation_set = validation_set

        # all for zeroshot in validation
        self.valid_meta_data = valid_meta_data
        self.distractor_features = distractor_features
        self.distractor_meta_data = distractor_meta_data
        self.property_one = property_one
        self.property_two = property_two

        if type(self.features) == type({}):
            self.keys = list(features.keys())

        if dataset is not None:
            self.obverter_setup = True

        if mean is None and type(features) == type({}):

            imgs = np.asarray([features[key].data
                                 for key in features])
            mean = np.mean(imgs, axis=0)
            std = np.std(imgs, axis=0)
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero

        elif mean is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero
        self.mean = mean
        self.std = std

        if not raw and not metadata:
            self.features = (features - self.mean) / (2 * self.std)

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor()
            ]
        )

    def __getitem__(self, indices):
        if type(self.features) == type({}):
            target_key = self.keys[indices[0]]
            target_img = self.features[target_key].data
            list_key = list(target_key)
            lkey = list_key[5]

            distractors = []

            # distractors = self.step3_distractors[target_key]
            # for i, d in enumerate(distractors):
            #     distractors[i] = self.transforms(d.data)
            for distractor_img in self.step3_distractors[target_key]:
                # distractor_img = self.step3_distractors[target_key]
                if self.raw:
                    distractor_img = self.transforms(distractor_img.data)
                distractors.append(distractor_img)
                if len(self.step3_distractors[target_key]) == 1:
                    distractors.append(distractor_img)

            if self.raw:# and not self.validation_set:
                target_img = self.transforms(target_img)

            # if self.validation_set:
            #     print(target_key)

            # print(len(distractors))
            return (target_img, distractors, indices, lkey)

                # print('train',len(distractors))
            # else:
            #     target_img = []
            #     for i in range(5):
            #         list_key[5] = str(i)
            #         class_key = ''.join(list_key)
            #         # class_key = target_key#'111111ab'
            #         # if i < 4:
            #         #     class_key = self.keys[indices[i]]
            #         # target_key = self.keys[np.random.randint(max(indices))]
            #         # list_key = list(target_key)
            #         # list_key[5] = str(np.random.randint(0,5))
            #         # target_key = ''.join(list_key)

            #         # class_key = target_key
            #         # if i == 2:
            #         #     class_key = target_key # testing related, remove when done
            #         class_distractors = []
            #         for distractor_img in self.step3_distractors[class_key]:
            #             if self.raw:
            #                 distractor_img = self.transforms(distractor_img.data)
            #             class_distractors.append(distractor_img)
            #         distractors.append(class_distractors)

            #         targ = self.features[target_key].data
            #         if self.raw:
            #             targ = self.transforms(targ)
            #         target_img.append(targ)

                # return (target_img, distractors, indices, lkey)
        else:
            # print(type(self.valid_meta_data) != type(None))
            # print(type(self.distractor_features) != type(None))
            # print(type(self.distractor_meta_data) != type(None))
            # stop
            # if statement for the zero-shot validation set
            if type(self.distractor_meta_data) != type(None):
                # print(self.property_one, self.property_two)
                # values to know which mini-one-hot vector should be variable
                one_begin = np.floor(self.property_one/3)*3
                one_end = np.floor(self.property_one/3)*3+3
                two_begin = np.floor(self.property_two/3)*3
                two_end = np.floor(self.property_two/3)*3+3
                # arange from 0 to num of properties (5) and remove zero-shot set values
                aranged = np.delete(np.arange(int(self.valid_meta_data.shape[1]/3)),np.s_[np.floor(self.property_one/3),np.floor(self.property_two/3)])

                # compare all other mini-one-hot vectors and take out only indices that match
                meta_target = np.r_['-1',self.valid_meta_data[indices[0],aranged[0]*3:aranged[0]*3+3],self.valid_meta_data[indices[0],aranged[1]*3:aranged[1]*3+3],self.valid_meta_data[indices[0],aranged[2]*3:aranged[2]*3+3]]
                meta_rest = np.r_['-1',self.distractor_meta_data[:,aranged[0]*3:aranged[0]*3+3],self.distractor_meta_data[:,aranged[1]*3:aranged[1]*3+3],self.distractor_meta_data[:,aranged[2]*3:aranged[2]*3+3]]
                # indexes of all features/metadata that matches target completely except in the two zero-shot values
                idxs = np.nonzero(np.all(meta_rest == meta_target,axis=1))[0]

                distractors_idxs = np.random.choice(idxs,3)

                vmd5 = self.convert_back(self.valid_meta_data[indices[0]])

                # distractors for validation zero-shot learning
                # distractors = []
                # for d_idx in distractors_idxs:
                #     distractor_img = self.distractor_features[d_idx]
                #     if self.raw:
                #         distractor_img = self.transforms(distractor_img)
                #     distractors.append(distractor_img)

            target_idx = indices[0]


            if type(self.distractor_meta_data) == type(None):
                distractors_idxs = indices[1:]
                # vmd5 = self.convert_back(self.valid_meta_data[indices[0]])
                vmd5 = []

            distractors = []
            for d_idx in distractors_idxs:
                if type(self.distractor_meta_data) != type(None):
                    # print('zs')
                    distractor_img = self.distractor_features[d_idx]
                else:
                    distractor_img = self.features[d_idx]
                if self.raw:
                    distractor_img = self.transforms(distractor_img)
                distractors.append(distractor_img)

            target_img = self.features[target_idx]

        if self.raw:# and not self.validation_set:
            target_img = self.transforms(target_img)

        # return (target_img, distractors, indices, 0)
        return (target_img, distractors, indices, vmd5)

    def __len__(self):
        if self.obverter_setup:
            return self.dataset.shape[0]
        else:
            if type(self.features) == type({}):
                print('Dataset size is',len(self.features))
                return len(self.features)
            else:
                print('Dataset size is',self.features.shape[0])
                return self.features.shape[0]

    def convert_back(self, vmd):
        vmd5 = torch.argmax(torch.reshape(torch.tensor(vmd),(5,3)),dim=1)
        return vmd5
        # return torch.chunk(torch.tensor(vmd),5)



    # def validation_target_meta(self):
    #     print(self.valid_meta_data.shape)
    #     return self.valid_meta_data
    # def validation_distractor_meta(self):
    #     return self.distractor_meta_data
