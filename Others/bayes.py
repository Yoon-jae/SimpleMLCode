# -*- coding: utf-8 -*-

def main():
    sensitivity = 0.8  # [0, 1] 암을 가지고 있을 때, 양성이라 판단 될 확률
    prior_prob = 0.004  # 총 인구를 기준으로 암을 가질 확률
    false_alarm = 0.1  # 병을 가지지 않지만 양성이라 판단 될 확률

    print("%.2lf%%" % (100 * mammogram_test(sensitivity, prior_prob, false_alarm)))


def mammogram_test(sensitivity, prior_prob, false_alarm):
    # A = 1 : 암으로 진단 되는 사건, B = 1 : 실제로 암을 가지고 있는 사건

    p_a1_b1 = sensitivity  # p(A = 1 | B = 1)
    p_b1 = prior_prob  # p(B = 1)
    p_b0 = 1 - p_b1  # p(B = 0)
    p_a1_b0 = false_alarm  # p(A = 1 | B = 0)
    p_a1 = p_a1_b0 * p_b0 + p_a1_b1 * p_b1  # p(A = 1) = p(A = 1 | B = 0) * p(B = 0) + P(A = 1 | B = 1) * p(B = 1)
    p_b1_a1 = p_a1_b1 * p_b1 / p_a1  # p(B = 1 | A = 1)

    return p_b1_a1


if __name__ == "__main__":
    main()
