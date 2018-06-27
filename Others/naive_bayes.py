# -*- coding: utf-8 -*-

def main():
    M1 = {'r': 0.7, 'g': 0.2, 'b': 0.1}
    M2 = {'r': 0.3, 'g': 0.4, 'b': 0.3}

    test = {'r': 4, 'g': 3, 'b': 3}

    print(naive_bayse(M1, M2, test, 0.7, 0.3))


def naive_bayse(M1, M2, test, M1_prior, M2_prior):
    M1_likelihood = M2_likelihood = 1

    for color, value in test.items():
        M1_likelihood *= M1[color] ** value
        M2_likelihood *= M2[color] ** value

    M1_post = M1_likelihood * M1_prior
    M2_post = M2_likelihood * M2_prior
    sum_post = M1_post + M2_post

    return [M1_post / sum_post, M2_post / sum_post]


if __name__ == "__main__":
    main()
