"""
Huffman 부호 실습 프로그램
- Huffman 트리 구성, 인코딩/디코딩, 평균 코드 길이·엔트로피·압축률 계산
"""

import heapq
import math
from collections import Counter
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Huffman 트리 노드
# ---------------------------------------------------------------------------

class Node:
    """Huffman 트리의 노드 (내부 노드 또는 리프)."""
    def __init__(self, symbol: str, freq: int, left=None, right=None):
        self.symbol = symbol   # 리프일 때만 의미 있음 (내부 노드는 None)
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: "Node") -> bool:
        """빈도로 비교 (heap에서 사용)."""
        return self.freq < other.freq

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# ---------------------------------------------------------------------------
# 빈도 계산 및 Huffman 트리 구성
# ---------------------------------------------------------------------------

def build_freq_table(data: str) -> Dict[str, int]:
    """문자열에서 각 문자의 빈도를 계산합니다."""
    return dict(Counter(data))


def build_huffman_tree(freq: Dict[str, int]) -> Node:
    """
    빈도 테이블로부터 Huffman 트리를 구성합니다.
    최소 힙을 사용해 빈도가 가장 작은 두 노드를 반복적으로 합칩니다.
    """
    if not freq:
        raise ValueError("빈도 테이블이 비어 있습니다.")
    # 리프 노드들: (빈도, 노드) — heap은 첫 번째 원소로 정렬
    heap: List[Tuple[int, Node]] = [
        (f, Node(sym, f)) for sym, f in freq.items()
    ]
    heapq.heapify(heap)
    # 노드가 하나만 있으면 그대로 반환
    if len(heap) == 1:
        (f, n) = heapq.heappop(heap)
        return Node(None, f, n, None)
    while len(heap) > 1:
        f1, n1 = heapq.heappop(heap)
        f2, n2 = heapq.heappop(heap)
        parent = Node(None, f1 + f2, n1, n2)
        heapq.heappush(heap, (parent.freq, parent))
    _, root = heapq.heappop(heap)
    return root


def build_code_table(root: Node) -> Dict[str, str]:
    """
    Huffman 트리에서 각 심볼(리프)에 대한 이진 코드워드를 구합니다.
    왼쪽 자식 = 0, 오른쪽 자식 = 1.
    """
    code_table: Dict[str, str] = {}

    def _walk(node: Node, path: str) -> None:
        if node.is_leaf():
            if node.symbol is not None:
                code_table[node.symbol] = path if path else "0"
            return
        if node.left:
            _walk(node.left, path + "0")
        if node.right:
            _walk(node.right, path + "1")

    _walk(root, "")
    return code_table


# ---------------------------------------------------------------------------
# 엔트로피 및 평균 코드 길이 (정보이론)
# ---------------------------------------------------------------------------

def entropy_from_freq(freq: Dict[str, int]) -> float:
    """빈도로부터 엔트로피 H(X) = -sum p_i log2(p_i) [bits]를 계산합니다."""
    total = sum(freq.values())
    if total <= 0:
        return 0.0
    probs = [f / total for f in freq.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def average_code_length(freq: Dict[str, int], code_table: Dict[str, str]) -> float:
    """평균 코드 길이 L = sum p_i * l_i [bits/symbol]를 계산합니다."""
    total = sum(freq.values())
    if total <= 0:
        return 0.0
    return sum(
        (freq[sym] / total) * len(code_table[sym])
        for sym in code_table if sym in freq
    )


# ---------------------------------------------------------------------------
# 인코딩 / 디코딩
# ---------------------------------------------------------------------------

def encode(text: str, code_table: Dict[str, str]) -> str:
    """문자열을 Huffman 부호 비트열(문자 '0','1')로 인코딩합니다."""
    return "".join(code_table[c] for c in text)


def decode(bit_string: str, root: Node) -> str:
    """비트열을 Huffman 트리를 따라 복호화합니다."""
    if not bit_string and root.is_leaf():
        return root.symbol or ""
    result = []
    node = root
    for bit in bit_string:
        node = node.left if bit == "0" else node.right
        if node is None:
            raise ValueError("잘못된 비트열입니다.")
        if node.is_leaf():
            result.append(node.symbol)
            node = root
    if node != root and not node.is_leaf():
        raise ValueError("비트열이 중간에서 끊겼습니다.")
    return "".join(result)


# ---------------------------------------------------------------------------
# 출력 및 예시
# ---------------------------------------------------------------------------

def print_code_table(freq: Dict[str, int], code_table: Dict[str, str]) -> None:
    """코드 테이블과 확률·코드 길이를 출력합니다."""
    total = sum(freq.values())
    print("  심볼 | 빈도   | 확률     | 코드워드  | 길이")
    print("  " + "-" * 45)
    for sym in sorted(code_table.keys(), key=lambda s: (-freq.get(s, 0), s)):
        f = freq.get(sym, 0)
        p = f / total if total else 0
        code = code_table[sym]
        print(f"  {sym!r:^4} | {f:5}  | {p:.4f}   | {code:8}  | {len(code)}")


def run_example(text: str) -> None:
    """주어진 문자열에 대해 Huffman 부호화 전체 과정을 실행하고 결과를 출력합니다."""
    print("=" * 60)
    print("입력 문자열:", repr(text))
    print("=" * 60)
    if not text:
        print("빈 문자열입니다.")
        return
    freq = build_freq_table(text)
    total = sum(freq.values())
    root = build_huffman_tree(freq)
    code_table = build_code_table(root)
    H = entropy_from_freq(freq)
    L_avg = average_code_length(freq, code_table)
    bits = encode(text, code_table)
    decoded = decode(bits, root)
    # 고정 길이 부호: ceil(log2(심볼 수)) 비트/심볼
    n_symbols = len(freq)
    fixed_bits = math.ceil(math.log2(n_symbols)) if n_symbols > 0 else 0
    L_fixed = fixed_bits
    original_bits = len(text) * 8  # ASCII 가정
    compressed_bits = len(bits)
    ratio = (compressed_bits / original_bits * 100) if original_bits else 0

    print("\n[1] 빈도 및 Huffman 코드 테이블")
    print_code_table(freq, code_table)
    print("\n[2] 정보이론량")
    print(f"  엔트로피 H(X)           = {H:.4f} bits/symbol")
    print(f"  Huffman 평균 코드 길이 L = {L_avg:.4f} bits/symbol")
    print(f"  고정 길이 부호 (심볼당)  = {L_fixed} bits/symbol")
    print(f"  관계: L >= H ? {L_avg >= H}  (L는 H에 근접 기대)")
    print("\n[3] 인코딩 결과")
    print(f"  비트열 (처음 80자): {bits[:80]}{'...' if len(bits) > 80 else ''}")
    print(f"  총 비트 수: {len(bits)}")
    print("\n[4] 압축 (문자열을 8비트/문자 가정)")
    print(f"  원본 비트 수:   {original_bits}")
    print(f"  압축 후 비트 수: {compressed_bits}")
    print(f"  압축률: {ratio:.1f}%")
    print("\n[5] 복호화 검증")
    print(f"  복원 문자열: {repr(decoded)}")
    print(f"  일치 여부: {decoded == text}")
    print()


def main() -> None:
    """실습 예시: 고정 문자열과 사용자 입력."""
    # 예시 1: 설명 자료의 예 (A~F 빈도와 유사한 분포)
    example1 = "A" * 45 + "B" * 13 + "C" * 12 + "D" * 16 + "E" * 9 + "F" * 5
    run_example(example1)
    # 예시 2: 짧은 문장
    example2 = "ABRACADABRA"
    run_example(example2)
    # 예시 3: 한글/영문 혼합 (동일 알고리즘 적용)
    example3 = "정보이론과 Huffman 부호"
    run_example(example3)


if __name__ == "__main__":
    main()
