"""
Data loading and statistics utilities for GoEmotions dataset
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

# Constants
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_LABELS = len(EMOTION_LABELS)
COLUMN_NAMES = ['text', 'emotion_ids', 'comment_id']


def load_dataset(tsv_path='dataset/data/'):
    """
    Load train, dev, and test datasets
    Returns tuple of (df_train, df_dev, df_test)
    """
    df_train = pd.read_csv(
        os.path.join(tsv_path, 'train.tsv'),
        sep='\t',
        header=None,
        names=COLUMN_NAMES,
        encoding='utf-8'
    )

    df_dev = pd.read_csv(
        os.path.join(tsv_path, 'dev.tsv'),
        sep='\t',
        header=None,
        names=COLUMN_NAMES,
        encoding='utf-8'
    )

    df_test = pd.read_csv(
        os.path.join(tsv_path, 'test.tsv'),
        sep='\t',
        header=None,
        names=COLUMN_NAMES,
        encoding='utf-8'
    )

    return df_train, df_dev, df_test


def create_binary_labels(df, labels_list):
    """
    Convert emotion_ids column to binary label matrix
    Returns DataFrame with binary columns for each emotion
    """
    label_matrix = pd.DataFrame(0, index=df.index, columns=labels_list)

    for index, row in df.iterrows():
        ids = str(row['emotion_ids']).split(',')

        try:
            int_ids = [int(i) for i in ids if i.isdigit()]
        except ValueError:
            int_ids = []

        for emotion_index in int_ids:
            if 0 <= emotion_index < NUM_LABELS:
                label_matrix.loc[index, labels_list[emotion_index]] = 1

    df_result = pd.concat([df[['text', 'comment_id']], label_matrix], axis=1)
    return df_result


def get_emotion_distribution(df):
    """
    Calculate emotion distribution from raw dataset
    Returns dictionary of emotion -> count
    """
    emotion_counts = Counter()

    for emotion_ids in df['emotion_ids']:
        ids = str(emotion_ids).split(',')
        try:
            int_ids = [int(i) for i in ids if i.isdigit()]
            for idx in int_ids:
                if 0 <= idx < NUM_LABELS:
                    emotion_counts[EMOTION_LABELS[idx]] += 1
        except ValueError:
            pass

    return dict(emotion_counts)


def get_multi_label_statistics(df):
    """
    Calculate multi-label statistics
    Returns dictionary with various multi-label metrics
    """
    labels_per_sample = Counter()
    total_labels = 0

    for emotion_ids in df['emotion_ids']:
        ids = str(emotion_ids).split(',')
        try:
            int_ids = [int(i) for i in ids if i.isdigit() and 0 <= int(i) < NUM_LABELS]
            num_labels = len(int_ids)
            labels_per_sample[num_labels] += 1
            total_labels += num_labels
        except ValueError:
            labels_per_sample[0] += 1

    avg_labels_per_sample = total_labels / len(df) if len(df) > 0 else 0

    # Calculate additional statistics
    max_labels = max(labels_per_sample.keys()) if labels_per_sample else 0
    samples_with_no_label = labels_per_sample.get(0, 0)
    samples_with_multiple_labels = sum(count for num_labels, count in labels_per_sample.items() if num_labels > 1)

    return {
        'labels_per_sample': dict(labels_per_sample),
        'avg_labels_per_sample': avg_labels_per_sample,
        'total_labels': total_labels,
        'max_labels': max_labels,
        'samples_with_no_label': samples_with_no_label,
        'samples_with_multiple_labels': samples_with_multiple_labels
    }


def get_text_statistics(df):
    """
    Calculate text statistics (length, vocabulary, etc.)
    Returns dictionary with text metrics
    """
    text_lengths = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()

    return {
        'avg_text_length': float(text_lengths.mean()),
        'min_text_length': int(text_lengths.min()),
        'max_text_length': int(text_lengths.max()),
        'avg_word_count': float(word_counts.mean()),
        'min_word_count': int(word_counts.min()),
        'max_word_count': int(word_counts.max())
    }


def get_dataset_statistics(df_train, df_dev, df_test):
    """
    Compute comprehensive dataset statistics
    Returns dictionary with all statistics
    """
    stats = {
        'splits': {
            'train': len(df_train),
            'dev': len(df_dev),
            'test': len(df_test),
            'total': len(df_train) + len(df_dev) + len(df_test)
        },
        'emotion_distribution': get_emotion_distribution(df_train),
        'multi_label_stats': get_multi_label_statistics(df_train),
        'text_stats': get_text_statistics(df_train)
    }

    # Calculate class imbalance ratio
    emotion_counts = stats['emotion_distribution']
    max_count = max(emotion_counts.values())
    min_count = min(emotion_counts.values())
    stats['imbalance_ratio'] = max_count / min_count if min_count > 0 else 0

    # Find most and least common emotions
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    stats['most_common_emotion'] = sorted_emotions[0][0]
    stats['least_common_emotion'] = sorted_emotions[-1][0]

    return stats


def get_sample_texts(df, n=5):
    """
    Get sample texts with their emotions
    Returns list of dictionaries with text and emotions
    """
    samples = []
    df_sample = df.sample(n=min(n, len(df)))

    for _, row in df_sample.iterrows():
        ids = str(row['emotion_ids']).split(',')
        try:
            int_ids = [int(i) for i in ids if i.isdigit() and 0 <= int(i) < NUM_LABELS]
            emotions = [EMOTION_LABELS[i] for i in int_ids]
        except ValueError:
            emotions = []

        samples.append({
            'text': row['text'],
            'emotions': emotions,
            'comment_id': row['comment_id']
        })

    return samples
