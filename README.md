# fake-news-detection
Machine learning models to detect fake news using data from [Kaggle](https://www.kaggle.com/c/fake-news/data) with [doc2vec](https://arxiv.org/abs/1405.4053), SVM, and [BERT](https://arxiv.org/abs/1810.04805).

# Install
git clone https://github.com/bubblemans/fake-news-detection.git
pip install -r requirements.txt

# Usage
We only use the text column from the original column, and we will drop other columns like title and author.
<br/>

For SVC, first, preprocess the text by removing punctuactions and lowcasing every token. Second, train doc2vec model to extract feature vectors from text. Last, train a SVM classifier using feature vectors and labels.

<br/>
For BERT, similar process is in one file. Please use Google Colab or other tools and run the script.

## Preprocess
```bash
python preprocess.py --raw_dir './data/train.csv' --train_dir './data/processed_train.csv' --eval_dir './data/processed_eval.csv'
```
## Train doc2vec
```bash
python train_doc2vec.py --train_dir './data/processed_train.csv' --vector_size 100 --output_model ./model/doc2vec.model
```

## Train SVM
```bash
python train_svm.py --data_dir './data/processed_train.csv' --input_model './model/doc2vec.model' --eval_dir './data/processed_eval.csv' --result './data/svm_result.csv'
```

## BERT
For BERT, please download the ipynb file and Kaggle data. Also, you need to modify the file path to where you store them in Google Drive.

```python
# example
train_df = pd.read_csv('/content/gdrive/MyDrive/data/train.csv')
```



