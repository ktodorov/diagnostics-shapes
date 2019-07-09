import numpy as np

from helpers.file_helper import FileHelper
from helpers.metadata_helper import get_shapes_metadata

from enums.dataset_type import DatasetType

file_helper = FileHelper()

def set_zero_shot_data(propertyOne, propertyTwo):
	train_features = np.load(file_helper.train_input_path)
	prev_train_len = train_features.shape[0]
	train_meta = get_shapes_metadata(dataset=DatasetType.Train)
	train_indices = [i for i,md in enumerate(train_meta) if not (md[propertyOne] == 1 and md[propertyTwo] == 1)]
	train_features = train_features[train_indices]
	print(f'Train features from {prev_train_len} to {train_features.shape[0]}')

	valid_features = np.load(file_helper.valid_input_path)
	prev_valid_len = valid_features.shape[0]
	valid_meta = get_shapes_metadata(dataset=DatasetType.Valid)
	valid_indices = [i for i,md in enumerate(valid_meta) if (md[propertyOne] == 1 and md[propertyTwo] == 1)]
	valid_features = valid_features[valid_indices]
	print(f'Valid features from {prev_valid_len} to {valid_features.shape[0]}')

	test_features = np.load(file_helper.test_input_path)
	prev_test_len = test_features.shape[0]
	test_meta = get_shapes_metadata(dataset=DatasetType.Test)
	test_indices = [i for i,md in enumerate(test_meta) if (md[propertyOne] == 1 and md[propertyTwo] == 1)]
	test_features = test_features[test_indices]
	print(f'Test features from {prev_test_len} to {test_features.shape[0]}')

	return train_features, valid_features, test_features

def set_zero_shot_meta(propertyOne, propertyTwo):
	train_meta = get_shapes_metadata(dataset=DatasetType.Train)
	prev_train_len = train_meta.shape[0]
	train_indices = [i for i,md in enumerate(train_meta) if not (md[propertyOne] == 1 and md[propertyTwo] == 1)]
	train_meta = train_meta[train_indices]
	print(f'Train meta from {prev_train_len} to {train_meta.shape[0]}')

	valid_meta = get_shapes_metadata(dataset=DatasetType.Valid)
	prev_valid_len = valid_meta.shape[0]
	valid_indices = [i for i,md in enumerate(valid_meta) if (md[propertyOne] == 1 and md[propertyTwo] == 1)]
	valid_meta = valid_meta[valid_indices]
	print(f'Valid meta from {prev_valid_len} to {valid_meta.shape[0]}')

	test_meta = get_shapes_metadata(dataset=DatasetType.Test)
	prev_test_len = test_meta.shape[0]
	test_indices = [i for i,md in enumerate(test_meta) if (md[propertyOne] == 1 and md[propertyTwo] == 1)]
	test_meta = test_meta[test_indices]
	print(f'Test meta from {prev_test_len} to {test_meta.shape[0]}')

	return train_meta, valid_meta, test_meta


### THESE ARE JUST USED FOR SOME TESTS
def take_out_single_property_DATA(propertyOne, propertyTwo):
	train_features = np.load(file_helper.train_input_path)
	prev_train_len = train_features.shape[0]
	train_meta = get_shapes_metadata(dataset=DatasetType.Train)
	train_indices = [i for i,md in enumerate(train_meta) if not (md[propertyOne] == 1 or md[propertyTwo] == 1)]
	train_features = train_features[train_indices]
	print(f'Train features from {prev_train_len} to {train_features.shape[0]}')

	valid_features = np.load(file_helper.valid_input_path)
	prev_valid_len = valid_features.shape[0]
	valid_meta = get_shapes_metadata(dataset=DatasetType.Valid)
	valid_indices = [i for i,md in enumerate(valid_meta) if (md[propertyOne] == 1 or md[propertyTwo] == 1)]
	valid_features = valid_features[valid_indices]
	print(f'Valid features from {prev_valid_len} to {valid_features.shape[0]}')

	test_features = np.load(file_helper.test_input_path)
	prev_test_len = test_features.shape[0]
	test_meta = get_shapes_metadata(dataset=DatasetType.Test)
	test_indices = [i for i,md in enumerate(test_meta) if (md[propertyOne] == 1 or md[propertyTwo] == 1)]
	test_features = test_features[test_indices]
	print(f'Test features from {prev_test_len} to {test_features.shape[0]}')

	return train_features, valid_features, test_features


def take_out_single_property_META(propertyOne, propertyTwo):
	train_meta = get_shapes_metadata(dataset=DatasetType.Train)
	prev_train_len = train_meta.shape[0]
	train_indices = [i for i,md in enumerate(train_meta) if not (md[propertyOne] == 1 or md[propertyTwo] == 1)]
	train_meta = train_meta[train_indices]
	print(f'Train meta from {prev_train_len} to {train_meta.shape[0]}')

	valid_meta = get_shapes_metadata(dataset=DatasetType.Valid)
	prev_valid_len = valid_meta.shape[0]
	valid_indices = [i for i,md in enumerate(valid_meta) if (md[propertyOne] == 1 or md[propertyTwo] == 1)]
	valid_meta = valid_meta[valid_indices]
	print(f'Valid meta from {prev_valid_len} to {valid_meta.shape[0]}')

	test_meta = get_shapes_metadata(dataset=DatasetType.Test)
	prev_test_len = test_meta.shape[0]
	test_indices = [i for i,md in enumerate(test_meta) if (md[propertyOne] == 1 or md[propertyTwo] == 1)]
	test_meta = test_meta[test_indices]
	print(f'Test meta from {prev_test_len} to {test_meta.shape[0]}')

	return train_meta, valid_meta, test_meta

if __name__ == "__main__":
	set_zero_shot_data(0, 3)

	take_out_single_property_DATA(0, 3)
