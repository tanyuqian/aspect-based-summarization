perl data_utils/gdown.pl https://drive.google.com/open?id=1goxgX-0_2Jo7cNAFrsb9BJTsrH7re6by cnn_stories_tokenized.tar.gz

tar -zxvf cnn_stories_tokenized.tar.gz

mv cnn_dm/val.source cnn_dm/dev.source
mv cnn_dm/val.target cnn_dm/dev.target

mkdir data
mv cnn_dm/ data/