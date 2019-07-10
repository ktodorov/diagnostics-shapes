import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from .metadata_helper import get_shapes_metadata
from .shape_helper import generate_shapes_dataset
from .feature_helper import get_features
from .file_helper import FileHelper

from datasets.shapes_dataset import ShapesDataset
from samplers.images_sampler import ImagesSampler

from enums.dataset_type import DatasetType

from generate_datasets import generate_step3_dataset

from data_to_zeroshot import *

file_helper = FileHelper()

def get_shapes_features(device, dataset=DatasetType.Valid, mode="features"):
    """
    Returns numpy array with matching features
    Args:
        dataset (str) in {'train', 'valid', 'test'}
        mode (str) in {"features", "raw"}
    """
    if mode == "features":
        features_path = file_helper.get_features_path(dataset)

        if not os.path.isfile(features_path):
            images = np.load(file_helper.get_input_path(dataset))

            features = get_features(images, device)
            np.save(features_path, features)
            assert len(features) == len(images)

        return np.load(features_path)
    else:
        images = np.load(file_helper.get_input_path(dataset))
        return images


def get_dataloaders(
    device,
    batch_size=16,
    k=3,
    debug=False,
    dataset="all",
    dataset_type="features",
    step3=False,
    zero_shot=False,
    property_one=0,
    property_two=3,
    valid_meta_data=None
):
    """
    Returns dataloader for the train/valid/test datasets
    Args:
        batch_size: batch size to be used in the dataloader
        k: number of distractors to be used in training
        debug (bool, optional): whether to use a much smaller subset of train data
        dataset (str, optional): whether to return a specific dataset or all
                                 options are {"train", "valid", "test", "all"}
                                 default: "all"
        dataset_type (str, optional): what datatype encoding to use: {"meta", "features", "raw"}
                                      default: "features"
    """
    if dataset_type == "raw":
        if not step3:
            if not zero_shot:
                train_features = np.load(file_helper.train_input_path)
                valid_features = np.load(file_helper.valid_input_path)
                test_features = np.load(file_helper.test_input_path)
                target_train_metadata = get_shapes_metadata(dataset=DatasetType.Train)
                target_valid_metadata = get_shapes_metadata(dataset=DatasetType.Valid)
            else:
                # train_features = np.load(file_helper.train_input_path)
                # valid_features = np.load(file_helper.valid_input_path)
                test_features = np.load(file_helper.test_input_path)
                print(f'Zero Shot on {property_one} and {property_two} from the metadata -- train_features removal')
                # train_features, valid_features, test_features = set_zero_shot_data(property_one, property_two)
                # train_features, valid_features, test_features = take_out_single_property_DATA(property_one, property_two)
                # print(valid_meta_data == valid_meta_data[1,:])
                train_features, valid_features, _ = set_zero_shot_data(property_one, property_two)
                target_train_metadata, target_valid_metadata, _ = set_zero_shot_meta(property_one, property_two)
                distractor_valid_features = np.load(file_helper.valid_input_path)

                # # print('Shapes of TARGET data')
                # # print(target_valid_features.shape, target_valid_metadata.shape)
                # # print('Shapes of ALL data')
                # # print(valid_features.shape,valid_meta_data.shape)

                # one_begin = round(property_one/3)*3
                # one_end = round(property_one/3)*3+3
                # two_begin = round(property_two/3)*3
                # two_end = round(property_two/3)*3+3

                # # print(valid_meta_data[one_begin],target_valid_metadata[0,:])
                # # print(np.delete(valid_meta_data, np.s_[one_begin:one_end], 1).shape)
                # # print(np.delete(target_valid_metadata[0,:],np.s_[one_begin:one_end],0).shape)

                # print('train_features',train_features.shape)
                # print('target valid_features',valid_features.shape)
                # print('distractor valid meta',valid_meta_data.shape)
                # print('distractor valid feat',distractor_valid_features.shape)

                # aranged = np.delete(np.arange(5),np.s_[round(property_one/3),round(property_two/3)])
                # meta_target = np.r_['-1',target_valid_metadata[0,aranged[0]*3:aranged[0]*3+3],target_valid_metadata[0,aranged[1]*3:aranged[1]*3+3],target_valid_metadata[0,aranged[2]*3:aranged[2]*3+3]]
                # meta_rest = np.r_['-1',valid_meta_data[:,aranged[0]*3:aranged[0]*3+3],valid_meta_data[:,aranged[1]*3:aranged[1]*3+3],valid_meta_data[:,aranged[2]*3:aranged[2]*3+3]]

                # idxs = np.nonzero(np.all(meta_rest == meta_target,axis=1))[0]
                # # print(idxs)
                # print(idxs.shape)
                # # print(np.random.choice(idxs))
                # # These valid_meta_data idx are the idx that link to targets who are identical except in the zero shot department
                # # for idx in idxs:
                # #     print(idx, valid_meta_data[idx,:], distractor_valid_features[idx].shape)

                # # note this has been done for first target_valid_metadata
                # print(target_valid_metadata[0,:], valid_features[0].shape, 0)
                # for i in range(3):
                #     idx = np.random.choice(idxs)
                #     print(valid_meta_data[idx,:], distractor_valid_features[idx].shape, idx)
                # # stop

                # idxs # indexes of distractors available in valid_meta_data

            train_dataset = ShapesDataset(train_features, valid_meta_data = target_train_metadata, raw=True)

            # All features are normalized with train mean and std
            
            if zero_shot:
                valid_dataset = ShapesDataset(
                    valid_features,
                    mean=train_dataset.mean,
                    std=train_dataset.std,
                    raw=True,
                    valid_meta_data=target_valid_metadata,
                    distractor_features=distractor_valid_features,
                    distractor_meta_data=valid_meta_data,
                    property_one=property_one,
                    property_two=property_two)
            else:
                valid_dataset = ShapesDataset(
                    valid_features,
                    mean=train_dataset.mean,
                    std=train_dataset.std,
                    raw=True)


            test_dataset = ShapesDataset(
                test_features,
                mean=train_dataset.mean,
                std=train_dataset.std,
                raw=True)

        else:
            # this part is hardcoded as it is currently only used for debugging
            randomized = False
            # zero_shot = False
            # property specific
            if not randomized:
                print('PROPERTY SPECIFIC')
                train_target_dict = pickle.load(open(file_helper.train_targets_path, 'rb'))
                train_distractors_dict = pickle.load(open(file_helper.train_distractors_path, 'rb'))

                valid_target_dict = pickle.load(open(file_helper.valid_targets_path, 'rb'))
                valid_distractors_dict = pickle.load(open(file_helper.valid_distractors_path, 'rb'))

                test_target_dict = pickle.load(open(file_helper.test_targets_path, 'rb'))
                test_distractors_dict = pickle.load(open(file_helper.test_distractors_path, 'rb'))
            # randomized
            else:
                print('RANDOMIZED')
                # train_target_dict = pickle.load(open(file_helper.train_targets_path, 'rb'))
                # train_distractors_dict = pickle.load(open(file_helper.train_distractors_path, 'rb'))

                if not zero_shot:
                    train_target_dict = pickle.load(open(file_helper.train_targets_RANDOM_path, 'rb'))
                    train_distractors_dict = pickle.load(open(file_helper.train_distractors_RANDOM_path, 'rb'))
                    valid_target_dict = pickle.load(open(file_helper.valid_targets_RANDOM_path, 'rb'))
                    valid_distractors_dict = pickle.load(open(file_helper.valid_distractors_RANDOM_path, 'rb'))

                else:
                    print('ZERO SHOT')
                    train_target_dict = pickle.load(open(file_helper.train_targets_zs1_path, 'rb'))
                    train_distractors_dict = pickle.load(open(file_helper.train_distractors_zs1_path, 'rb'))
                    print(len(train_target_dict))
                    valid_target_dict = pickle.load(open(file_helper.valid_targets_zs1_path, 'rb'))
                    valid_distractors_dict = pickle.load(open(file_helper.valid_distractors_zs1_path, 'rb'))

                # valid_target_dict = pickle.load(open(file_helper.valid_targets_path, 'rb'))
                # valid_distractors_dict = pickle.load(open(file_helper.valid_distractors_path, 'rb'))

                test_target_dict = pickle.load(open(file_helper.test_targets_RANDOM_path, 'rb'))
                test_distractors_dict = pickle.load(open(file_helper.test_distractors_RANDOM_path, 'rb'))

            train_dataset = ShapesDataset(
                train_target_dict, 
                step3_distractors = train_distractors_dict, 
                raw=True)
            
            valid_dataset = ShapesDataset(
                valid_target_dict,
                mean=train_dataset.mean,
                std=train_dataset.std,
                step3_distractors = valid_distractors_dict,
                raw=True,
                validation_set = True)

            test_dataset = ShapesDataset(
                test_target_dict,
                mean=train_dataset.mean,
                std=train_dataset.std,
                step3_distractors = test_distractors_dict,
                raw=True)

    if dataset_type == "features":

        train_features = get_shapes_features(device, dataset=DatasetType.Train)
        valid_features = get_shapes_features(device, dataset=DatasetType.Valid)
        test_features = get_shapes_features(device, dataset=DatasetType.Test)

        if debug:
            train_features = train_features[:10000]

        train_dataset = ShapesDataset(train_features)

        # All features are normalized with train mean and std
        valid_dataset = ShapesDataset(
            valid_features,
            mean=train_dataset.mean,
            std=train_dataset.std)

        test_dataset = ShapesDataset(
            test_features,
            mean=train_dataset.mean,
            std=train_dataset.std)

    if dataset_type == "meta":
        train_meta = get_shapes_metadata(dataset=DatasetType.Train)
        valid_meta = get_shapes_metadata(dataset=DatasetType.Valid)
        test_meta = get_shapes_metadata(dataset=DatasetType.Test)

        train_dataset = ShapesDataset(
            train_meta.astype(np.float32), metadata=True)
        valid_dataset = ShapesDataset(
            valid_meta.astype(np.float32), metadata=True)
        test_dataset = ShapesDataset(
            test_meta.astype(np.float32), metadata=True)

    train_data = DataLoader(
        train_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(train_dataset, k, shuffle=True),
            batch_size=batch_size,
            drop_last=False,
        ),
    )

    valid_data = DataLoader(
        valid_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(valid_dataset, k, shuffle=False),
            batch_size=batch_size,
            drop_last=False,
        ),
    )

    test_data = DataLoader(
        test_dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ImagesSampler(test_dataset, k, shuffle=False),
            batch_size=batch_size,
            drop_last=False,
        ),
    )

    if dataset == "train":
        return train_data
    if dataset == "valid":
        return valid_data
    if dataset == "test":
        return test_data
    else:
        return train_data, valid_data, test_data


def get_shapes_dataloader(
        device,
        batch_size=16,
        k=3,
        debug=False,
        dataset="all",
        dataset_type="features",
        step3=False,
        zero_shot=False,
        property_one=0,
        property_two=3,
        valid_meta_data=None):
    """
    Args:
        batch_size (int, opt): batch size of dataloaders
        k (int, opt): number of distractors
    """

    if not os.path.exists(file_helper.train_features_path):
        print("Features files not present - generating dataset")
        if not step3:
            pass
            # generate_shapes_dataset()
        else:
            # for step 3, generate different pickle file
            # note that generate_shapes_dataset creates two files
            # one for img.metadata and one for img.data
            # generate_property_set creates one, with imgs itself
            pass
            # generate_step3_dataset()

    return get_dataloaders(
        device,
        batch_size=batch_size,
        k=k,
        debug=debug,
        dataset=dataset,
        dataset_type=dataset_type,
        step3=step3,
        zero_shot=zero_shot,
        property_one=property_one,
        property_two=property_two,
        valid_meta_data=valid_meta_data)

