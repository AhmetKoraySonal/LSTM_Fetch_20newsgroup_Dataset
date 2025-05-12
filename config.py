# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:54:23 2025

@author: koray
"""
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="LSTM Model Training")

    # Argümanlar
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_words', type=int, default=10000, help='Maximum number of words for Tokenizer')
    parser.add_argument('--maxlen', type=int, default=100, help='Maximum length of padded sequences')
    parser.add_argument('--output_dim', type=int, default=32, help='Dimension of word embeddings')
    parser.add_argument('--lstm_units', type=int, default=32, help='Number of units in LSTM layer')

    args = parser.parse_args()
    return args