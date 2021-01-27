import os
import fire
import pickle

from data_utils.datasets import WeakSupDataset, MANewsDataset, get_dataloaders
from models.bart import BART

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from loguru import logger


BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 10
LR = 3e-5
NUM_WORKERS = 6
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.05
MAX_SRC_LENGTH = 1024
MAX_TGT_LENGTH = 140
LABEL_SMOOTH = 0.1
PRECISION = 16
N_WIKI_WORDS = 10
MODEL_INIT = 'bart.large.cnn'


def get_dataset(dataset_name, train_docs, wiki_sup):
    if dataset_name == 'weaksup':
        return {split: WeakSupDataset(
            split=split, n_docs=train_docs if split == 'train' else None)
            for split in ['train', 'dev']}
    elif dataset_name == 'manews':
        supervisor = pickle.load(open('supervisions/supervisor.pickle', 'rb')) \
            if wiki_sup else None
        return {split: MANewsDataset(
            split=split, n_docs=train_docs if split == 'train' else None,
            supervisor=supervisor, n_wiki_words=N_WIKI_WORDS)
            for split in ['train', 'dev']}
    else:
        raise ValueError(f'dataset {dataset_name} is not existed. '
                         f'it should be either \'weaksup\' or \'manews\'.')


def main(dataset_name='weaksup', n_epochs=1, train_docs=100000,
         pretrained_ckpt=None, wiki_sup=True):
    dataset = get_dataset(
        dataset_name=dataset_name, train_docs=train_docs, wiki_sup=wiki_sup)

    print(f'Data Examples:')
    print(dataset['train'][0], '\n', '=' * 100)
    print(dataset['train'][1], '\n', '=' * 100)

    dataloaders = get_dataloaders(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        dynamic_shape=True,
        max_src_length=MAX_SRC_LENGTH,
        max_tgt_length=MAX_TGT_LENGTH,
        shuffle=True)

    if pretrained_ckpt is not None:
        log_dir = f'logs/{dataset_name}_plus/docs{train_docs}/'
    else:
        log_dir = f'logs/{dataset_name}/docs{train_docs}/'

    if os.path.exists(log_dir):
        print(f'log_dir \"{log_dir}\" exists. training skipped.')
        return
    os.makedirs(log_dir)

    logger.add(f'{log_dir}/log.txt')
    logger.info(f'pretrained checkpoint: {pretrained_ckpt}')
    if pretrained_ckpt is None:
        bart = BART(init=MODEL_INIT, text_logger=logger)
    else:
        bart = BART.load_from_checkpoint(
            checkpoint_path=pretrained_ckpt,
            init=MODEL_INIT, text_logger=logger)

    train_steps = n_epochs * (
            len(dataloaders['train']) // ACCUMULATE_GRAD_BATCHES + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    bart.set_hparams(
        batch_size=BATCH_SIZE,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON,
        label_smooth=LABEL_SMOOTH)

    checkpoint_callback = ModelCheckpoint(
        filepath=f'{log_dir}/best_model',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True)

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        max_epochs=n_epochs,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gpus=1,
        precision=PRECISION)

    trainer.fit(
        model=bart,
        train_dataloader=dataloaders['train'],
        val_dataloaders=dataloaders['dev'])

    logger.info('training finished.')


if __name__ == '__main__':
    fire.Fire(main)