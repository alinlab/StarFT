from ast import arg
import os
import sys
from datetime import datetime
import glob
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import copy
import time
from tqdm import trange
try:
    import wandb
except ImportError:
    wandb = None

from clip.loss import ClipLoss, StarReg, SpuriousKLReg

from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder
from src.models.utils import cosine_lr
from src.models.zeroshot import get_zeroshot_classifier

from src.args import parse_arguments

import src.datasets as datasets
from src.datasets.data import get_data


def setup_logging(log_file, level):
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

def get_latest_checkpoint(path):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints)
        return checkpoints[-1]
    return None

def main(args):
    args = parse_arguments(args)

    # Get the name of the experiments
    args.resume = None
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        spurious_type = args.spurious_path.split('/')[-1].split('.')[0]\
            if args.spurious_path is not None else 'none'
        descriptor_type = args.descriptor_path.split('/')[-1].split('.')[0]\
            if args.descriptor_path is not None else 'none'
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = f"{args.train_dataset}/{args.exp_name}/"
        args.name += '-'.join([
            f"spurious_{spurious_type}",
            f"bs_{args.batch_size}",
            f"wd_{args.wd}",
            f"lr_{args.lr}",
            f"seed_{args.seed}",
            f"loss_{args.loss_type}",
            f"type_{args.spurious_type}",
            date_str,
        ])

        if not args.keep_lang:
            args.name += f"-keep_lang_{str(args.keep_lang)}"
        
        if args.reg_ratio is not None:
            args.name += f"-ratio_{str(args.reg_ratio)}"

        if args.diminishing:
            logging.info("Using diminishing!!")
            args.name += f"-dim"

    else:
        args.resume=True

    args.save = os.path.join(args.save, args.name) # args.save = "./logs/" + args.name = "ImageNet/exp_name/name"
    os.makedirs(args.save, exist_ok=True)

    # Setup text logger
    args.log_path = os.path.join(args.save, "log.log")
    setup_logging(args.log_path, logging.INFO)
    assert args.save is not None, 'Please provide a path to store models'

    # Setup wandb, tensorboard, checkpoint logging
    if args.wandb:
        args.wandb = 'wandb'
    args.checkpoint_path = os.path.join(args.save, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True # slow down speeds
    # torch.backends.cudnn.benchmark = False

    # Initialize the CLIP encoder
    model = CLIPEncoder(args, keep_lang=args.keep_lang) # clip_encoder
    classification_head = ClassificationHead(normalize=True, weights=None)
    
    # Log model and parameters
    if args.train_dataset:
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.save, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")


    # Initialize model
    model = model.cuda()
    classification_head = classification_head.cuda()
    train_preprocess = model.train_preprocess
    val_preprocess = model.val_preprocess
    model.process_images = True
    print_every = 100


    # Specify device
    devices = list(range(torch.cuda.device_count()))
    logging.info('Using devices' + str(devices))
    model = torch.nn.DataParallel(model, device_ids=devices)
    if args.loss_type == "ce":
        classification_head = get_zeroshot_classifier(args, model.module.model)
        classification_head = classification_head.cuda()
    classification_head = torch.nn.DataParallel(classification_head,
                                                device_ids=devices)
    if len(devices) > 1:
        args.distributed = True


    # Specify loss function & optimizer
    if args.loss_type == "ce":
        clip_loss_fn = torch.nn.CrossEntropyLoss()

    elif args.loss_type == "contrastive":
        clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)
    if args.spurious_path and args.reg_ratio:
        assert args.spurious_type is not None, "Please provide a regularization type."
        if args.spurious_type == "kl":
            reg_fn = SpuriousKLReg()
        elif args.spurious_type == "star":
            reg_fn = StarReg()
        else:
            raise NotImplementedError(f"Regularization not implemented for {args.spurious_type}")

    clip_params = list(model.parameters())
    total_params = clip_params
    if args.loss_type == "ce":
        total_params.extend(list(classification_head.parameters()))
    params = [p for p in total_params if p.requires_grad]


    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    model_orig = copy.deepcopy(model)


    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        logging.info('Resume training...')
        resume_from = None
        checkpoint_path = args.checkpoint_path
        resume_from = get_latest_checkpoint(checkpoint_path)
        if resume_from:
            logging.info(f'Found latest resume checkpoint at {resume_from}.')
        else:
            logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')

        with open(resume_from, "rb") as f:
            checkpoint = torch.load(f, map_location='cpu')

        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info(f"=> resuming checkpoint '{resume_from}' (epoch {start_epoch})")
 
    # Initialize dataset & scheduler
    assert args.train_dataset is not None, "Please provide a training dataset."
    logging.info(f"Fine-tuning using {args.train_dataset}")
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(train_preprocess, 
                            location=args.data_location, 
                            batch_size=args.batch_size)
    
    img_text_data = get_data(
        args, (train_preprocess, val_preprocess), epoch=start_epoch
    )

    assert len(img_text_data), 'At least one train or eval dataset must be specified.'
 
    ft_dataloader = img_text_data['train_ft'].dataloader
    ft_iterator = iter(ft_dataloader)
    num_batches = len(dataset.train_loader)

    logging.info(f"Num batches is {num_batches}")

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                          args.epochs * num_batches, args.min_lr)
    
    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = ft_dataloader.num_samples

        # you will have to configure this for your project!
        wandb.init(
            project=args.exp_name,
            name=args.name.split("/")[-1],
            id=args.name.split("/")[-1],
            notes="",
            tags=[],
            resume='auto' if args.resume else None,
            config=vars(args),
        )
        # if args.debug:
        #     wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    model.train()
    classification_head.train()
    model_orig.eval()

    stats = []
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Epoch : {epoch}")
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_flyp_loss_sum = 0

        reg_ratio = args.reg_ratio
        if args.diminishing and args.spurious_type in ["star", "kl"]:
            reg_ratio = args.reg_ratio * (1 - (epoch+1) / args.epochs)
            logging.info(f"Using reg_ratio {reg_ratio}, diminishing...")
    

        for i in trange(num_batches):
            start_time = time.time()
            step = i + epoch * num_batches
            if epoch != -1:
                scheduler(step)
            optimizer.zero_grad()

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                ft_iterator = iter(ft_dataloader)
                ft_batch = next(ft_iterator)


            if args.spurious_path and args.reg_ratio:
                ft_image, ft_text, label, ft_spurious = ft_batch
                ft_image, ft_text, ft_spurious = ft_image.cuda(), ft_text.cuda(), ft_spurious.cuda()
            else:
                ft_image, ft_text, label = ft_batch
                ft_image, ft_text = ft_image.cuda(), ft_text.cuda()


            ft_image_features, ft_text_features, logit_scale = model(ft_image, ft_text)


            if args.distributed:
                logit_scale = logit_scale[0]

            if args.loss_type == "ce":
                label = label.cuda()
                logit = classification_head(ft_image_features)
                ft_clip_loss = clip_loss_fn(logit, label)

            elif args.loss_type == "contrastive":
                ft_clip_loss = clip_loss_fn(ft_image_features, ft_text_features, logit_scale)
                
            if args.spurious_path and args.reg_ratio:

                ft_text_features_spurious = model(None, ft_spurious)
                ft_text_features_spurious = F.normalize(ft_text_features_spurious, dim=1)

                with torch.no_grad():
                    ft_image_features_spurious_orig, ft_text_features_spurious_orig, logit_scale_orig = model_orig(ft_image, ft_spurious)
                    
                if args.distributed:
                    logit_scale_orig = logit_scale_orig[0]
                

                if args.spurious_type == "star":
                    label = label.cuda()
                    reg_term = reg_fn(ft_image_features_spurious_orig, ft_text_features_spurious_orig,
                                      ft_image_features, ft_text_features_spurious,
                                      logit_scale_orig,
                                      label=label)
                else:
                    reg_term = reg_fn(ft_image_features_spurious_orig, ft_text_features_spurious_orig, ft_image_features, ft_text_features_spurious, logit_scale_orig, logit_scale)
                ft_clip_loss = ft_clip_loss + reg_ratio * reg_term

            ft_clip_loss.backward()
            optimizer.step()


            id_flyp_loss_sum += ft_clip_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                info = f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\tID FLYP Loss: {ft_clip_loss.item():.4f}"
                if args.spurious_path and args.reg_ratio:
                    info += f"\tREG: {reg_term.item():.4f}"
                logging.info(info)

                log_data = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "ID FLYP Loss": ft_clip_loss.item(),
                    "Avg ID FLYP Loss": (id_flyp_loss_sum / (i+1)),
                    "scale": logit_scale.item(),
                }

                if args.spurious_path and args.reg_ratio:
                    log_data.update({"REG": reg_term.item()})

                for name, val in log_data.items():
                    name = "train/" + name
                    if args.wandb:
                        assert wandb is not None, 'Please install wandb.'
                        wandb.log({name: val, 'step': step})


        id_flyp_loss_avg = id_flyp_loss_sum / num_batches

        # Evaluate
        classification_head_new = get_zeroshot_classifier(args, model.module.model)
        classification_head_new = classification_head_new.cuda()

        eval_results = evaluate(model, args, classification_head_new,
                                epoch_stats)

        completed_epoch = epoch + 1
        # Saving model
        if args.save is not None:
            if args.epochs <= 10:
                if args.loss_type == "ce":
                    checkpoint_dict = {
                        "epoch": completed_epoch,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "classification_head": classification_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                elif args.loss_type == "contrastive":
                    checkpoint_dict = {
                        "epoch": completed_epoch,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))

                logging.info(f'Saving model to {args.checkpoint_path}')
            else:
                if completed_epoch % 10 == 0:
                    if args.loss_type == "ce":
                        checkpoint_dict = {
                            "epoch": completed_epoch,
                            "name": args.name,
                            "state_dict": model.state_dict(),
                            "classification_head": classification_head.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                    elif args.loss_type == "contrastive":
                        checkpoint_dict = {
                            "epoch": completed_epoch,
                            "name": args.name,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                    torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))

                    logging.info(f'Saving model to {args.checkpoint_path}')   

        ood_acc = 0
        num_datasets = 0
        for k, v in epoch_stats.items():
            if 'Accuracy' in k:
                if k == 'ImageNet Accuracy':
                    #ignore the ID acc term
                    continue
                ood_acc += v
                num_datasets += 1
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
        else:
            ood_acc = 0

        epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        logging.info(f"Avg OOD Acc : {ood_acc:.4f}")
        logging.info(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")
        epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
       
        if args.resume:
            epoch_df = pd.DataFrame(epoch_stats, index=list(map(int, str(epoch))))
            epoch_df.to_csv(os.path.join(args.save, "stats.tsv"), mode='a', header=False, sep='\t')
        else:
            stats_df.to_csv(os.path.join(args.save, "stats.tsv"), sep='\t')

    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main(sys.argv[1:])
