import fire
import pickle
import cleantext
from tqdm import trange

from data_utils.datasets import MANewsDataset
from models.bart import BART


BATCH_SIZE = 12
N_WIKI_WORDS = 10
MAX_LEN = 140
MIN_LEN = 55
LEN_PENALTY = 2.
NO_REPEAT_NGRAM_SIZE = 3
BEAM_SIZE = 4
MODEL_INIT = 'bart.large.cnn'


def main(log_path, wiki_sup=True):
    supervisor = pickle.load(open('supervisions/supervisor.pickle', 'rb')) \
        if wiki_sup else None
    dataset = MANewsDataset(
        split='test', supervisor=supervisor, n_wiki_words=N_WIKI_WORDS)
    test_examples = [example for example in dataset]

    bart = BART.load_from_checkpoint(
        init=MODEL_INIT,
        checkpoint_path=f'{log_path}/best_model.ckpt').to('cuda')
    bart.eval()

    src_file = open(f'{log_path}/test.source', 'w')
    gold_file = open(f'{log_path}/test.gold', 'w')
    hypo_file = open(f'{log_path}/test.hypo', 'w')

    for i in trange(0, len(test_examples), BATCH_SIZE, desc=f'Generating'):
        batch_examples = test_examples[i:i+BATCH_SIZE]

        gen_texts = bart.generate(
            src_texts=[example['src'] for example in batch_examples],
            max_len=MAX_LEN,
            min_len=MIN_LEN,
            beam_size=BEAM_SIZE,
            len_penalty=LEN_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE)

        for example, gen_text in zip(batch_examples, gen_texts):
            print(example['src'].replace('\n\n', ' ||| '), file=src_file)
            print(example['tgt'], file=gold_file)
            print(cleantext.clean(gen_text, extra_spaces=True, lowercase=True),
                  file=hypo_file)


if __name__ == '__main__':
    fire.Fire(main)