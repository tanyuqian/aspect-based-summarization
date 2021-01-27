import os

cache_dir = f'{os.getenv("HOME")}/.cache/'
if not os.path.exists(f'{cache_dir}/stanford-corenlp-full-2016-10-31.zip'):
    os.system(f'wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip -P {cache_dir}')
    os.system(f'unzip {cache_dir}/stanford-corenlp-full-2016-10-31.zip -d {cache_dir}')
else:
    print(f'{cache_dir}/stanford-corenlp-full-2016-10-31.zip already existed.')
