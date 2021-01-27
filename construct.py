import os
import fire
import json
from tqdm import tqdm
import pickle

from supervisions.supervisor import Supervisor


N_WIKI_WORDS = 10


def load_data(split, n_examples):
    src_file = open(f'data/cnn_dm/{split}.source')
    tgt_file = open(f'data/cnn_dm/{split}.target')

    documents, summaries = [], []
    for src, tgt in zip(src_file.readlines()[:n_examples],
                        tgt_file.readlines()[:n_examples]):
        documents.append(src.strip())
        summaries.append(tgt.strip())

    return documents, summaries


def main(split):
    train_documents, _ = load_data(split='train', n_examples=None)
    supervisor = Supervisor()
    supervisor.build_tfidf_vectorizer(documents=train_documents)
    pickle.dump(supervisor, open('supervisions/supervisor.pickle', 'wb'))
    print('supervisor initialized.')

    documents, summaries = load_data(split=split, n_examples=None)

    for l in range(0, len(documents), 10000):
        r = min(len(documents), l + 10000)

        if os.path.exists(f'{split}_{l}-{r}.json'):
            continue

        dataset = []
        json.dump(dataset, open(f'{split}_{l}-{r}.json', 'w'), indent=4)

        prog_bar = tqdm(zip(documents[l:r], summaries[l:r]),
                        total=len(documents[l:r]))
        for doc_id, (document, summary) in enumerate(prog_bar):
            in_text_entities, related_entities = \
                supervisor.guess_aspects(summary)

            dataset.append({
                'doc_id': doc_id,
                'document': document,
                'global_summary': summary,
                'aspect_summaries': []
            })

            prog_bar.set_postfix_str(
                f'in_text_ents: {len(in_text_entities)}, '
                f'rel_ents: {len(related_entities)}')
            for aspect in in_text_entities + related_entities:
                guessed_summary = supervisor.guess_summary(
                    aspect=aspect, global_summary=summary)

                wiki_words = supervisor.get_wiki_words(
                    aspect=aspect, document=document, n_limit=N_WIKI_WORDS)

                if guessed_summary is not None:
                    dataset[-1]['aspect_summaries'].append({
                        'aspect': aspect,
                        'summary': guessed_summary['aspect_summary'],
                        'reasonings': guessed_summary['reasonings'],
                        'wiki_words': wiki_words
                    })

            json.dump(dataset, open(f'{split}_{l}-{r}.json', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
