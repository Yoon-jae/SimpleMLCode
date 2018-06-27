# -*- coding: utf-8 -*-

import re

special_chars_remover = re.compile("[^\w'|_]")


def main():
    sentence = "The Asia-Pacific Case Competition challenges students to critically evaluate the fast-evolving landscape of artificial intelligence. How we can harness the full potential of this promising technology? How do we effectively mitigate the risks?"
    bow = create_BOW(sentence)

    print(bow)


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


def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)


if __name__ == "__main__":
    main()
