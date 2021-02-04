import fire
import pickle
import json
import cleantext

from models.bart import BART


N_WIKI_WORDS = 10
MAX_LEN = 140
MIN_LEN = 55
LEN_PENALTY = 2.
NO_REPEAT_NGRAM_SIZE = 3
BEAM_SIZE = 4
MODEL_INIT = 'bart.large.cnn'


def main(ckpt_path, wiki_sup=True):
    supervisor = pickle.load(open('supervisions/supervisor.pickle', 'rb')) \
        if wiki_sup else None

    bart = BART.load_from_checkpoint(
        init=MODEL_INIT, checkpoint_path=ckpt_path).to('cuda')
    bart.eval()

    demo_input = json.load(open('demo_input.json'))

    document, aspects = demo_input['document'], demo_input['aspects']
    document = cleantext.clean(document, extra_spaces=True, lowercase=True)

    print('=' * 50)
    print('DOCUMENT:', document)
    print('=' * 50)

    for aspect in aspects:
        wiki_words = supervisor.get_wiki_words(
            aspect=aspect, document=document, n_limit=N_WIKI_WORDS) \
            if supervisor is not None else []

        src = '{aspect} : {wiki_words}\n\n{document}'.format(
            aspect=aspect.lower(),
            wiki_words=' '.join(wiki_words),
            document=document)

        gen_text = bart.generate(
            src_texts=[src],
            max_len=MAX_LEN,
            min_len=MIN_LEN,
            beam_size=BEAM_SIZE,
            len_penalty=LEN_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE)[0]

        print('-' * 50)
        print('ASPECT:', aspect)
        print('-' * 50)
        print('GENERATED SUMMARY:', gen_text)
        print('=' * 50)


if __name__ == '__main__':
    fire.Fire(main)