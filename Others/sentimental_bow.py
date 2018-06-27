# -*- coding: utf-8 -*-

import matplotlib as mpl

mpl.use("Agg")
import re
import math

special_chars_remover = re.compile("[^\w'|_]")


def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)


def main():
    training_sentences = read_data()
    testing_sentence = "어설픈 연기들로 몰입이 전혀 안되네요"

    prob_pair = naive_bayes(training_sentences, testing_sentence)

    print(testing_sentence)
    print(prob_pair)


def naive_bayes(training_sentences, testing_sentence):
    log_prob_negative = calculate_doc_prob(training_sentences[0], testing_sentence, 0.1) + math.log(0.5)
    log_prob_positive = calculate_doc_prob(training_sentences[1], testing_sentence, 0.1) + math.log(0.5)
    prob_pair = normalize_log_prob(log_prob_negative, log_prob_positive)

    return prob_pair


def read_data():
    training_sentences = [[], []]

    with open('./ratings.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines[1:]):
            ID, document, label = line.split('\t')

            if label == '0\n':
                training_sentences[0].append(document)
            elif label == '1\n':
                training_sentences[1].append(document)

    return [' '.join(training_sentences[0]), ' '.join(training_sentences[1])]


def normalize_log_prob(prob1, prob2):
    maxprob = max(prob1, prob2)

    prob1 -= maxprob
    prob2 -= maxprob
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)

    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant
    prob2 *= normalize_constant

    return (prob1, prob2)


def calculate_doc_prob(training_sentence, testing_sentence, alpha):
    logprob = 0

    training_model = create_BOW(training_sentence)
    testing_model = create_BOW(testing_sentence)

    total_token = 0
    for word, freq in training_model.items():
        total_token += freq

    for word, freq in testing_model.items():
        if word not in training_model:
            logprob += math.log(alpha / total_token)
        else:
            logprob += math.log(training_model[word] / total_token) * freq

    return logprob


def create_BOW(sentence):
    bow = {}

    sentence_lowered = sentence.lower()
    sentence_without_special_characters = remove_special_characters(sentence_lowered)
    splitted_sentence = sentence_without_special_characters.split()
    splitted_sentence_filtered = [
        token
        for token in splitted_sentence if len(token) >= 1
    ]

    for token in splitted_sentence_filtered:
        bow.setdefault(token, 0)
        bow[token] += 1

    return bow


if __name__ == "__main__":
    main()
