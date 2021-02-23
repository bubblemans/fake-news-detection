import pandas as pd
import numpy as np
import argparse

from gensim.models.doc2vec import Doc2Vec
from sklearn.svm import SVC


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--eval_dir', required=True)
    parser.add_argument('--result', required=True)
    return parser.parse_args()


def load_data(data_dir='./data/processed_train.csv', input_model='./model/doc2vec.model'):
     # training dataset
    df = pd.read_csv(data_dir)
    df.fillna('', inplace=True)

    train_texts = df['text'].tolist()
    train_labels = df['labels'].tolist()
    # for text in train_texts:
    #     print(text)
    #     if type(text) == float:
            # print(text)

    # load doc2vec model
    d2v_model = Doc2Vec.load(input_model)
    train_vec = [d2v_model.infer_vector(text.split()) for text in train_texts]

    return d2v_model, train_vec, train_labels


def train(vector, labels):
    svm_clf = SVC()
    svm_clf.fit(vector, labels)
    return svm_clf


def evaluate(d2v_model, svc_model, eval_dir='./data/processed_eval.csv', result='./data/svm_result.csv'):
    df = pd.read_csv(eval_dir)
    df.fillna('', inplace=True)
    texts = df['text'].tolist()
    labels = df['labels'].tolist()
    test_t2c = [d2v_model.infer_vector(text.split()) for text in texts]

    # evaluate
    pred_labels = svc_model.predict(test_t2c)
    tp = fp = fn = tn = 0
    for pred, actual in zip(pred_labels, labels):
        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 1 and pred == 0:
            fn += 1
        elif actual == 0 and pred == 0:
            tn += 1
    print('tp:', tp, 'fp:', fp, 'tn:', tn, 'fn:', fn)
    print('accuray:', (tp + tn) / (tp + fp + fn + tn))

    dict_to_df = {'predict': pred_labels, 'actual': labels}
    df = pd.DataFrame(dict_to_df)
    df.to_csv(result)


if __name__ == '__main__':
    args = _parse_args()
    d2v_model, vector, labels = load_data(args.data_dir, args.input_model)
    svc_model = train(vector, labels)
    evaluate(d2v_model, svc_model, eval_dir='./data/processed_eval.csv', result='./data/svm_result.csv')