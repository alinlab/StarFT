import os
import torch
from tqdm import tqdm

import numpy as np

import clip.clip as clip

import src.templates as templates
import src.datasets as datasets

from src.args import parse_arguments
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.eval import evaluate

from open_clip import get_tokenizer



def get_zeroshot_classifier(args, clip_model):
    assert args.template is not None
    assert args.train_dataset is not None
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale

    few_shot_data_list = ["ImageNetKShot", "PatchCamelyonVal"]
    dataset_class = getattr(datasets, args.train_dataset)
    if args.train_dataset in few_shot_data_list:
        # assert args.k != None
        # assert args.k != 0
        print(f"Doing {args.k} shot classification")
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size,
                                k=args.k)
    else:
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size)
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            
            # FIXME tokenizer hack
            if args.model == "ViT-B-16-SigLIP":
                tokenizer = get_tokenizer(args.model)    
                texts = tokenizer(texts).to(device)
            else:            
                texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(
                texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head

def get_spurious_classifier(args, clip_model):
    assert args.spurious_template is not None
    assert args.train_dataset is not None
    template = getattr(templates, args.spurious_template)
    logit_scale = clip_model.logit_scale

    few_shot_data_list = ["ImageNetKShot", "PatchCamelyonVal"]
    dataset_class = getattr(datasets, args.train_dataset)
    if args.train_dataset in few_shot_data_list:
        # assert args.k != None
        # assert args.k != 0
        print(f"Doing {args.k} shot classification")
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size,
                                k=args.k)
    else:
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size)
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            
            # FIXME tokenizer hack
            if args.model == "ViT-B-16-SigLIP":
                tokenizer = get_tokenizer(args.model)    
                texts = tokenizer(texts).to(device)
            else:            
                texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(
                texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head

def eval(args):
    args.freeze_encoder = True
    if args.load is not None:
        classifier = ImageClassifier.load(args.load)
    else:
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args,
                                                      image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder,
                                     classification_head,
                                     process_images=False)

    evaluate(classifier, args)

    if args.save is not None:
        classifier.save(args.save)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)