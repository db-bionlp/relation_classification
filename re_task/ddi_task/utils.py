import random
import logging
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import numpy as np
from transformers import AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from re_task.ddi_task.model import Bert_for_re
import sys
from re_task.SemEval2010_task.official_eval import official_f1
from collections import Counter

MODEL_CLASSES = {
    'bert': (BertConfig, Bert_for_re, BertTokenizer),
    'roberta': (RobertaConfig, Bert_for_re, RobertaTokenizer),
    'albert': (AlbertConfig, Bert_for_re, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': '../../resources/biobert_v1.1_pubmed',
    'roberta': 'roberta-base',
    'albert': 'albert-xxlarge-v1'}


def get_label(args):
    return [label.strip() for label in open(args.label_file, 'r', encoding='utf-8')]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def score(key, prediction, file, verbose=True):

    NO_RELATION = "DDI-false"
    # NO_RELATION = "negative"
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        # print("Per-relation statistics:")
        writer = open(file, "a", encoding='utf-8')
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            writer.write(("{:<" + str(longest_relation) + "}").format(relation))
            writer.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(' ')
                writer.write(' ')
            if prec < 1.0:
                sys.stdout.write(' ')
                writer.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            writer.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            writer.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(' ')
                writer.write(' ')
            if recall < 1.0:
                sys.stdout.write(' ')
                writer.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            writer.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            writer.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(' ')
                sys.stdout.write(' ')
            if f1 < 1.0:
                sys.stdout.write(' ')
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
            writer.write("{:.2%}".format(f1))
            writer.write("  #: %d" % gold)
            writer.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    result = []
    result.append(prec_micro)
    result.append(recall_micro)
    result.append(f1_micro)

    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))

    return result



def acc_and_f1(preds, labels, file):

    acc = simple_accuracy(labels, preds)
    P = precision_score(labels, preds, average='macro')
    R = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    # label = [label.strip() for label in open('../dataset/SemEval2010_task8_corpus/label.txt', 'r', encoding='utf-8')]
    # new_preds = []
    # for pre in preds:
    #     a = label[int(pre)]
    #     new_preds.append(a)
    #
    # new_labels = []
    # for la in labels:
    #     b = label[int(la)]
    #     new_labels.append(b)
    #
    # result = score(new_labels, new_preds,file)

    return {
        'P': P,
        'R': R,
        'acc': acc,
        'f1': f1,
        # 'Precision(micro)': result[0],
        # 'Recall (micro)': result[1],
        # 'F1 (micro)': result[2]

    }

def compute_metrics(preds, labels, file):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels, file)


