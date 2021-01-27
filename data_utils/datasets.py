import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BartTokenizer


def get_dataloaders(dataset, batch_size, num_workers, dynamic_shape,
                    max_src_length, max_tgt_length, shuffle):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    def _collate_fn(raw_batch):
        hf_batch = tokenizer.prepare_seq2seq_batch(
            src_texts=[example['src'] for example in raw_batch],
            tgt_texts=[example['tgt'] for example in raw_batch],
            max_length=max_src_length, max_target_length=max_tgt_length,
            padding='max_length' if not dynamic_shape else 'longest',
            return_tensors='pt')

        fs_batch = {
            'src_tokens': hf_batch['input_ids'],
            'src_lengths': torch.sum(hf_batch['attention_mask'], dim=1),
            'prev_output_tokens': hf_batch['labels']
        }

        return fs_batch

    return {
        split: DataLoader(
            dataset=dataset[split],
            batch_size=batch_size,
            collate_fn=_collate_fn,
            shuffle=(split == 'train' and shuffle),
            drop_last=True,
            num_workers=num_workers)
        for split in ['train', 'dev']}


class WeakSupDataset(Dataset):
    def __init__(self, split, n_docs=None):
        data_path = f'data/weaksup/{split}.json'
        if not os.path.exists(data_path):
            os.system('bash data_utils/download_weaksup.sh')

        docs = json.load(open(data_path))
        docs = docs[:n_docs]

        self._examples = []
        for doc in docs:
            for asp_sum in doc['aspect_summaries']:
                self._examples.append({
                    'aspect': asp_sum['aspect'],
                    'wiki_words': asp_sum['wiki_words'],
                    'document': doc['document'],
                    'summary': asp_sum['summary']
                })

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        example = self._examples[item]
        return {
            'src': '{aspect} : {wiki_words}\n\n{doc}'.format(
                aspect=example['aspect'],
                wiki_words=' '.join(example['wiki_words']),
                doc=example['document']),
            'tgt': example['summary']
        }


class MANewsDataset(Dataset):
    def __init__(self, split, supervisor, n_wiki_words, n_docs=None):
        data_path = f'data/manews/{split}.json'
        if not os.path.exists(data_path):
            os.system('bash data_utils/download_manews.sh')

        self._examples = json.load(open(data_path))[:n_docs]
        for example in tqdm(self._examples, desc=f'loading manews {split} set'):
            example['wiki_words'] = supervisor.get_wiki_words(
                aspect=example['aspect'], document=example['document'],
                n_limit=n_wiki_words) if supervisor is not None else []

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        example = self._examples[item]
        return {
            'src': '{aspect} : {wiki_words}\n\n{doc}'.format(
                aspect=example['aspect'],
                wiki_words=' '.join(example['wiki_words']),
                doc=example['document']),
            'tgt': example['summary']
        }
