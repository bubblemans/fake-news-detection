import pandas as pd
import argparse

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--vector_size', type=int, default=100, required=False)
    parser.add_argument('--output_model', required=True)
    return parser.parse_args()


# python train_doc2vec.py --train_dir './data/processed_train.csv' --vector_size 100 --output_model ./model/doc2vec.model
if __name__ == '__main__':
    args = _parse_args()
    train_file = args.train_dir
    df = pd.read_csv(train_file)

    texts = df['text'].tolist()
    labels = df['labels'].tolist()

    documents = [TaggedDocument(str(doc), [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=args.vector_size, min_count=2, epochs=10, workers=4)
    model.save(args.output_model)
