import torch
import numpy as np

from data.agent_vocab import AgentVocab
from .metadata_helper import get_shapes_metadata, get_metadata_properties
from .dataloader_helper import get_shapes_features, get_shapes_dataloader

from enums.dataset_type import DatasetType

from models.shapes_receiver import ShapesReceiver
from models.shapes_sender import ShapesSender
from models.shapes_trainer import ShapesTrainer
from models.shapes_single_model import ShapesSingleModel
from models.shapes_meta_visual_module import ShapesMetaVisualModule
from models.messages_receiver import MessagesReceiver


def get_sender_receiver(device, args) -> (ShapesSender, ShapesReceiver, MessagesReceiver):
    # Load Vocab
    vocab = AgentVocab(args.vocab_size)

    baseline_receiver = None
    diagnostic_receiver = None

    cell_type = "lstm"
    genotype = {}
    if args.single_model:
        sender = ShapesSingleModel(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            greedy=args.greedy,
            cell_type=cell_type,
            genotype=genotype,
            dataset_type=args.dataset_type,
        )

        baseline_receiver = ShapesSingleModel(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            greedy=args.greedy,
            cell_type=cell_type,
            genotype=genotype,
            dataset_type=args.dataset_type,
        )
    else:
        sender = ShapesSender(
            args.vocab_size,
            args.max_length,
            vocab.bound_idx,
            device,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            greedy=args.greedy,
            cell_type=cell_type,
            genotype=genotype,
            dataset_type=args.dataset_type,
            inference_step=args.inference_step
        )

        if not args.inference_step or args.multi_task:
            baseline_receiver = ShapesReceiver(
                args.vocab_size,
                device,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )

    if args.inference_step or args.multi_task:
        diagnostic_receiver = MessagesReceiver(
            num_classes_by_model=[3, 3, 2, 3, 3],
            device=device)

    if args.sender_path:
        sender = torch.load(args.sender_path)
    if args.receiver_path:
        baseline_receiver = torch.load(args.receiver_path)

    # This is only used when not training using raw data
    # if args.freeze_sender:
    #     for param in sender.parameters():
    #         param.requires_grad = False
    # else:
    #     s_visual_module = ShapesMetaVisualModule(
    #         hidden_size=sender.hidden_size, dataset_type=args.dataset_type
    #     )
    #     sender.input_module = s_visual_module

    # if args.freeze_receiver:
    #     for param in receiver.parameters():
    #         param.requires_grad = False
    # else:
    #     r_visual_module = ShapesMetaVisualModule(
    #         hidden_size=receiver.hidden_size,
    #         dataset_type=args.dataset_type,
    #         sender=False,
    #     )

    #     if args.single_model:
    #         receiver.output_module = r_visual_module
    #     else:
    #         receiver.input_module = r_visual_module

    return sender, baseline_receiver, diagnostic_receiver


def get_trainer(
        sender,
        device,
        inference_step,
        multi_task,
        multi_task_lambda,
        dataset_type,
        step3,
        baseline_receiver=None,
        diagnostic_receiver=None,
        disabled_properties=None):
    extract_features = dataset_type == "raw"

    return ShapesTrainer(
        sender,
        device,
        inference_step,
        multi_task,
        multi_task_lambda,
        step3,
        baseline_receiver=baseline_receiver,
        diagnostic_receiver=diagnostic_receiver,
        extract_features=extract_features,
        disabled_properties=disabled_properties)


def get_meta_data():
    train_meta_data = get_metadata_properties(dataset=DatasetType.Train)
    valid_meta_data = get_metadata_properties(dataset=DatasetType.Valid)
    test_meta_data = get_metadata_properties(dataset=DatasetType.Test)
    return train_meta_data, valid_meta_data, test_meta_data


def get_training_data(device, batch_size, k, debugging, dataset_type, step3, zero_shot, property_one, property_two):

    valid_meta_data = get_shapes_metadata(dataset=DatasetType.Valid)
    valid_features = get_shapes_features(device=device, dataset=DatasetType.Valid)

    # Load data
    train_data, valid_data, test_data = get_shapes_dataloader(
        device=device,
        batch_size=batch_size,
        k=k,
        debug=debugging,
        dataset_type=dataset_type,
        step3=step3,
        zero_shot=zero_shot,
        property_one=property_one,
        property_two=property_two,
        valid_meta_data=valid_meta_data)

    # valid_meta_data = get_shapes_metadata(dataset=DatasetType.Valid)
    valid_features = get_shapes_features(
        device=device, dataset=DatasetType.Valid)

    return (train_data, valid_data, test_data, valid_meta_data, valid_features)


def get_raw_data(args, dataset=DatasetType.Valid):
    if args.task == "shapes":
        valid_raw = get_shapes_features(dataset=dataset, mode="raw")
        return valid_raw
    else:
        raise ValueError(
            "Unsupported task type for raw : {}".formate(args.task))


def save_example_images(args, filename):
    if args.save_example_batch:
        valid_raw = get_raw_data(args)
        valid_raw = valid_raw[:10]
        file_path = filename + "/example_batch.npy"
        np.save(file_path, valid_raw)
