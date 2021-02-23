#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import argparse
import string
import pandas as pd

from sklearn.model_selection import train_test_split


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--eval_dir', required=True)
    return parser.parse_args()


def text_cleaning(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()


def preprocess(input_dir, test_size, train_dir='./data/processed_train.csv', eval_dir='./data/processed_eval.csv'):
    df = pd.read_csv(input_dir)
    df = df.drop(columns=['id', 'title', 'author'])
    df.fillna('', inplace=True)
    df['text'] = df['text'].apply(text_cleaning)
    df.columns = ['text', 'labels']
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df.to_csv(train_dir)
    test_df.to_csv(eval_dir)


# python preprocess_all_data.py --raw_dir './data/train.csv' --train_dir './data/processed_train.csv' --eval_dir './data/processed_eval.csv'
if __name__ == '__main__':
    args = _parse_args()
    preprocess(args.raw_dir, 0.15, train_dir=args.train_dir, eval_dir=args.eval_dir)
    print('done')
