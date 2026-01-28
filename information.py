"""정보이론 유틸리티: 자기 정보량, 엔트로피, 결합/조건부 엔트로피, 상호 정보량 계산."""

import math
from typing import List, Sequence


## probability: float 는 probability가 실수형임을 알려주는 type hint입니다.
## -> float는 반환 값이 실수형이라고 알려주는 type hint입니다.

def self_information(probability: float) -> float:
    """
    자기 정보량을 계산하는 함수

    Args:
        probability: 사건의 확률 (0 < p <= 1)

    Returns:
        자기 정보량 (bits)
    """
    if probability <= 0 or probability > 1:
        raise ValueError("확률은 0과 1 사이의 값이어야 합니다.")
    return -math.log2(probability)


def entropy(probabilities: Sequence[float]) -> float:
    """
    엔트로피를 계산하는 함수

    Args:
        probabilities: 확률 분포 (리스트 또는 시퀀스)

    Returns:
        엔트로피 (bits)
    """
    return float(-sum(p * math.log2(p) for p in probabilities if p > 0))


def mutual_information(
    joint_prob: List[List[float]],
    x_prob: List[float],
    y_prob: List[float],
) -> float:
    """
    상호 정보량을 계산하는 함수

    Args:
        joint_prob: 결합 확률 P(X, Y) (2D 리스트)
        x_prob: X의 주변 확률 P(X) (1D 리스트)
        y_prob: Y의 주변 확률 P(Y) (1D 리스트)

    Returns:
        상호 정보량 (bits)
    """
    mi = 0.0
    for i in range(len(x_prob)):
        for j in range(len(y_prob)):
            p_xy = joint_prob[i][j]
            if p_xy > 0:
                p_x, p_y = x_prob[i], y_prob[j]
                if p_x > 0 and p_y > 0:
                    mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def joint_entropy(joint_prob: List[List[float]]) -> float:
    """
    결합 엔트로피를 계산하는 함수

    Args:
        joint_prob: 결합 확률 P(X, Y) (2D 리스트)

    Returns:
        결합 엔트로피 (bits)
    """
    flat = [p for row in joint_prob for p in row]
    return entropy(flat)


def conditional_entropy(
    joint_prob: List[List[float]],
    given_prob: List[float],
) -> float:
    """
    조건부 엔트로피 H(X|Y)를 계산하는 함수

    Args:
        joint_prob: 결합 확률 P(X, Y) (2D 리스트, 행=X, 열=Y)
        given_prob: 조건 변수 Y의 주변 확률 P(Y) (1D 리스트)

    Returns:
        조건부 엔트로피 H(X|Y) (bits)
    """
    h_cond = 0.0
    for j in range(len(given_prob)):
        if given_prob[j] > 0:
            col = [joint_prob[i][j] for i in range(len(joint_prob))]
            cond_prob = [v / given_prob[j] for v in col]
            h_cond += given_prob[j] * entropy(cond_prob)
    return h_cond


def _transpose(matrix: List[List[float]]) -> List[List[float]]:
    """2D 리스트 전치."""
    if not matrix:
        return []
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def main() -> None:
    """정보이론 함수들 예시 실행."""
    # 예시 1: 자기 정보량 계산
    print("=== 자기 정보량 예시 ===")
    print(f"동전 앞면 (P=0.5): {self_information(0.5):.3f} bits")
    print(f"주사위 6 (P=1/6): {self_information(1/6):.3f} bits")
    print(f"확실한 사건 (P=1.0): {self_information(1.0):.3f} bits")
    print(f"드문 사건 (P=0.01): {self_information(0.01):.3f} bits")
    print()

    # 예시 2: 상호 정보량 계산
    print("=== 상호 정보량 예시 ===")
    joint_indep = [[0.25, 0.25], [0.25, 0.25]]
    x_prob_indep = [0.5, 0.5]
    y_prob_indep = [0.5, 0.5]
    mi_indep = mutual_information(joint_indep, x_prob_indep, y_prob_indep)
    print(f"독립적인 변수: I(X;Y) = {mi_indep:.3f} bits")

    joint_dep = [[0.5, 0.0], [0.0, 0.5]]
    x_prob_dep = [0.5, 0.5]
    y_prob_dep = [0.5, 0.5]
    mi_dep = mutual_information(joint_dep, x_prob_dep, y_prob_dep)
    print(f"완전 의존 변수: I(X;Y) = {mi_dep:.3f} bits")

    joint_partial = [[0.4, 0.1], [0.1, 0.4]]
    x_prob_partial = [0.5, 0.5]
    y_prob_partial = [0.5, 0.5]
    mi_partial = mutual_information(joint_partial, x_prob_partial, y_prob_partial)
    print(f"부분 의존 변수: I(X;Y) = {mi_partial:.3f} bits")
    print()

    # 예시 3: 엔트로피, 결합/조건부 엔트로피
    print("=== 엔트로피 예시 ===")
    fair_coin = [0.5, 0.5]
    print(f"공정한 동전: H = {entropy(fair_coin):.3f} bits")
    biased_coin = [0.9, 0.1]
    print(f"편향된 동전 (0.9/0.1): H = {entropy(biased_coin):.3f} bits")
    fair_dice = [1 / 6] * 6
    print(f"공정한 주사위: H = {entropy(fair_dice):.3f} bits")
    print()

    print("=== 결합 엔트로피와 조건부 엔트로피 ===")
    joint_example = [[0.4, 0.1], [0.1, 0.4]]
    x_prob_example = [0.5, 0.5]
    y_prob_example = [0.5, 0.5]
    h_x_ex = entropy(x_prob_example)
    h_y_ex = entropy(y_prob_example)
    h_xy_ex = joint_entropy(joint_example)
    h_x_given_y = conditional_entropy(joint_example, y_prob_example)
    joint_T = _transpose(joint_example)
    h_y_given_x = conditional_entropy(joint_T, x_prob_example)
    print(f"H(X) = {h_x_ex:.3f} bits, H(Y) = {h_y_ex:.3f} bits")
    print(f"H(X,Y) = {h_xy_ex:.3f} bits")
    print(f"H(X|Y) = {h_x_given_y:.3f} bits, H(Y|X) = {h_y_given_x:.3f} bits")
    print()
    print("=== 관계식 검증 ===")
    print(f"I(X;Y) = H(X) - H(X|Y) = {h_x_ex - h_x_given_y:.3f} bits")
    print(f"I(X;Y) = H(X) + H(Y) - H(X,Y) = {h_x_ex + h_y_ex - h_xy_ex:.3f} bits")
    print(f"연쇄 법칙: H(X,Y) = H(X) + H(Y|X) = {h_x_ex + h_y_given_x:.3f} bits")


if __name__ == "__main__":
    main()
