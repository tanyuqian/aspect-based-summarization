log_path=$1

hypo_path=${log_path}/test.hypo
ref_path=${log_path}/test.gold

export CLASSPATH=$HOME/.cache/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

# Tokenize hypothesis and target files.
cat $hypo_path | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
cat $ref_path | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target

#bert-score -r test.hypo.target -c test.hypo.tokenized --lang en --rescale-with-baseline > ${log_path}/bert_scores.txt &

files2rouge test.hypo.tokenized test.hypo.target > ${log_path}/rouge_scores.txt &
files2rouge test.hypo.tokenized test.hypo.target

# Expected output: (ROUGE-2 Average_F: 0.21238)