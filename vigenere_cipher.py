"""
비즈네르 암호(Vigenère Cipher) 실습 프로그램
- 암호화/복호화
- 배비지-카시스키 공격: 반복 구절 분석, 일치 지수(IC), 빈도 분석

Vigenere_암호_설명자료.md와 함께 사용하세요.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A_ORD = ord("A")

# 영어 글자 빈도 (대략적 순서, ETAOIN SHRDLCU...)
ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"

# 영어 단일 문자 기대 빈도 (%)
ENGLISH_FREQ = {
    "E": 12.70, "T": 9.06, "A": 8.17, "O": 7.51, "I": 6.97, "N": 6.75,
    "S": 6.33, "H": 6.09, "R": 5.99, "D": 4.25, "L": 4.03, "C": 2.78,
    "U": 2.76, "M": 2.41, "W": 2.36, "F": 2.23, "G": 2.02, "Y": 1.97,
    "P": 1.93, "B": 1.29, "V": 0.98, "K": 0.77, "J": 0.15, "X": 0.15,
    "Q": 0.10, "Z": 0.07,
}


# ---------------------------------------------------------------------------
# 유틸리티: 알파벳만 추출
# ---------------------------------------------------------------------------

def _to_upper_letters(text: str) -> str:
    """텍스트에서 알파벳만 추출하여 대문자로 반환합니다."""
    return "".join(c.upper() for c in text if c.upper() in ALPHABET)


# ---------------------------------------------------------------------------
# 암호화 / 복호화
# ---------------------------------------------------------------------------

def encrypt(plaintext: str, key: str) -> str:
    """
    비즈네르 암호로 암호화합니다.
    C_i = (P_i + K_i) mod 26
    """
    plain = _to_upper_letters(plaintext)
    key_clean = _to_upper_letters(key)
    if not key_clean:
        raise ValueError("키는 최소 1글자 이상이어야 합니다.")
    result = []
    for i, p in enumerate(plain):
        k = key_clean[i % len(key_clean)]
        p_val = ord(p) - A_ORD
        k_val = ord(k) - A_ORD
        c_val = (p_val + k_val) % 26
        result.append(chr(A_ORD + c_val))
    return "".join(result)


def decrypt(ciphertext: str, key: str) -> str:
    """
    비즈네르 암호를 복호화합니다.
    P_i = (C_i - K_i) mod 26
    """
    cipher = _to_upper_letters(ciphertext)
    key_clean = _to_upper_letters(key)
    if not key_clean:
        raise ValueError("키는 최소 1글자 이상이어야 합니다.")
    result = []
    for i, c in enumerate(cipher):
        k = key_clean[i % len(key_clean)]
        c_val = ord(c) - A_ORD
        k_val = ord(k) - A_ORD
        p_val = (c_val - k_val) % 26
        result.append(chr(A_ORD + p_val))
    return "".join(result)


# ---------------------------------------------------------------------------
# 배비지-카시스키 공격: 반복 구절 찾기
# ---------------------------------------------------------------------------

def find_repeated_sequences(text: str, min_len: int = 3, max_len: int = 6) -> Dict[str, List[int]]:
    """
    암호문에서 반복되는 구절(3~6글자)과 그 위치들을 찾습니다.
    반환: {구절: [위치1, 위치2, ...]}
    """
    text = _to_upper_letters(text)
    positions: Dict[str, List[int]] = {}
    for length in range(min_len, max_len + 1):
        for i in range(len(text) - length + 1):
            seq = text[i : i + length]
            if seq not in positions:
                positions[seq] = []
            positions[seq].append(i)
    # 2번 이상 나타나는 구절만 반환
    return {seq: pos for seq, pos in positions.items() if len(pos) >= 2}


def kasiski_analysis(ciphertext: str) -> List[Tuple[int, int]]:
    """
    카시스키 검사: 반복 구절 간 거리의 GCD를 구해 키 길이 후보를 추정합니다.
    반환: [(키길이후보, 점수), ...] - 점수가 높을수록 후보 가능성 큼
    """
    repeats = find_repeated_sequences(ciphertext)
    if not repeats:
        return [(i, 0) for i in range(1, 16)]

    distances: List[int] = []
    for seq, positions in repeats.items():
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = positions[j] - positions[i]
                if d > 0:
                    distances.append(d)

    if not distances:
        return [(i, 0) for i in range(1, 16)]

    # 각 키 길이 k에 대해: 여러 거리가 k의 배수인 개수를 점수로
    key_length_scores: Dict[int, int] = {}
    for k in range(2, 21):
        score = sum(1 for d in distances if d % k == 0)
        key_length_scores[k] = score

    # 점수 내림차순 정렬
    sorted_candidates = sorted(
        key_length_scores.items(),
        key=lambda x: (-x[1], x[0])
    )
    return sorted_candidates[:10]


# ---------------------------------------------------------------------------
# 일치 지수 (Index of Coincidence)
# ---------------------------------------------------------------------------

def index_of_coincidence(text: str) -> float:
    """
    일치 지수 IC = sum n_i(n_i-1) / (N(N-1))
    무작위: ~0.038, 영어: ~0.065
    """
    text = _to_upper_letters(text)
    if len(text) < 2:
        return 0.0
    counts = Counter(c for c in text if c in ALPHABET)
    n = sum(counts.values())
    if n < 2:
        return 0.0
    ic = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    return ic


def ic_key_length_estimate(ciphertext: str, max_key_len: int = 15) -> List[Tuple[int, float]]:
    """
    여러 키 길이 후보에 대해 암호문을 열로 나누고,
    각 열의 IC 평균이 영어에 가까운 k를 찾습니다.
    반환: [(키길이, 평균IC), ...] - IC가 0.065에 가까울수록 후보
    """
    cipher = _to_upper_letters(ciphertext)
    results: List[Tuple[int, float]] = []
    for k in range(1, max_key_len + 1):
        columns = [""] * k
        for i, c in enumerate(cipher):
            if c in ALPHABET:
                columns[i % k] += c
        ics = [index_of_coincidence(col) for col in columns if len(col) > 1]
        avg_ic = sum(ics) / len(ics) if ics else 0.0
        results.append((k, avg_ic))
    # IC가 0.065(영어)에 가까운 순으로 정렬
    english_ic = 0.065
    results.sort(key=lambda x: -1 / (1 + abs(x[1] - english_ic)))
    return results


# ---------------------------------------------------------------------------
# 빈도 분석으로 키 문자 추정
# ---------------------------------------------------------------------------

def frequency_analysis(text: str) -> Dict[str, float]:
    """텍스트에서 각 문자의 출현 비율을 계산합니다."""
    text = _to_upper_letters(text)
    total = sum(1 for c in text if c in ALPHABET)
    if total == 0:
        return {c: 0.0 for c in ALPHABET}
    counts = Counter(c for c in text if c in ALPHABET)
    return {c: counts.get(c, 0) / total * 100 for c in ALPHABET}


def find_best_shift(cipher_col: str) -> Tuple[int, str]:
    """
    단일 치환된 열에서 가장 그럴듯한 시프트(키 문자)를 찾습니다.
    상관관계를 최대화하는 시프트를 선택합니다.
    반환: (시프트값 0~25, 추정 키문자)
    """
    freq = frequency_analysis(cipher_col)
    best_shift = 0
    best_score = -1.0
    for shift in range(26):
        # cipher_col의 문자들을 -shift 해서 "평문"으로 본 뒤
        # 영어 빈도와의 상관관계 계산
        score = 0.0
        for c in ALPHABET:
            # 암호문 c가 평문 (c - shift)에 해당
            plain_char = chr(A_ORD + (ord(c) - A_ORD - shift) % 26)
            score += freq.get(c, 0) * ENGLISH_FREQ.get(plain_char, 0)
        if score > best_score:
            best_score = score
            best_shift = shift
    key_char = chr(A_ORD + best_shift)
    return (best_shift, key_char)


def crack_vigenere(ciphertext: str, key_length: int) -> str:
    """
    키 길이를 알고 있을 때, 빈도 분석으로 키를 복원합니다.
    """
    cipher = _to_upper_letters(ciphertext)
    columns = [""] * key_length
    for i, c in enumerate(cipher):
        if c in ALPHABET:
            columns[i % key_length] += c
    key_chars = []
    for col in columns:
        _, k = find_best_shift(col)
        key_chars.append(k)
    return "".join(key_chars)


# ---------------------------------------------------------------------------
# 자동 공격: 키 길이 추정 + 키 복원
# ---------------------------------------------------------------------------

def auto_attack(ciphertext: str) -> Tuple[str, str, int]:
    """
    암호문만으로 키를 추정하고 평문을 복원합니다.
    반환: (추정키, 복원평문, 사용한 키길이)
    """
    # 1) 카시스키와 IC를 결합해 키 길이 추정
    kasiski = kasiski_analysis(ciphertext)
    ic_est = ic_key_length_estimate(ciphertext)
    # 카시스키 상위 5개와 IC 상위 5개에서 가중 평균
    candidates: Dict[int, float] = {}
    for rank, (k, score) in enumerate(kasiski[:5]):
        weight = 2.0 / (1 + rank)  # 1위 가중치 높음
        candidates[k] = candidates.get(k, 0) + weight * (score + 1)
    for rank, (k, ic) in enumerate(ic_est[:5]):
        dist = abs(ic - 0.065)
        weight = 1.0 / (1 + dist + rank * 0.1)  # IC가 영어에 가까울수록, 순위 높을수록
        candidates[k] = candidates.get(k, 0) + weight
    # 동점이면 짧은 키 길이 선호 (단순한 키가 더 흔함)
    best_key_len = max(candidates.keys(), key=lambda x: (candidates[x], -x))
    # 2) 키 복원
    key = crack_vigenere(ciphertext, best_key_len)
    plain = decrypt(ciphertext, key)
    return (key, plain, best_key_len)


# ---------------------------------------------------------------------------
# 데모 및 대화형 모드
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """설명과 함께 기본 예제를 실행합니다."""
    print("=" * 60)
    print("비즈네르 암호 실습 - '해독 불가능'이었던 암호")
    print("=" * 60)
    print("\n[역사적 배경]")
    print("  비즈네르 암호는 16세기부터 'le chiffre indechiffrable'")
    print("  (해독 불가능)이라 불렸습니다. 1846년경 찰스 배비지가,")
    print("  1863년 프리드리히 카시스키가 독립적으로 공격법을 발견했습니다.")
    print()

    # 예제 1: 기본 암호화/복호화
    plain1 = "ATTACKATDAWN"
    key1 = "LEMON"
    cipher1 = encrypt(plain1, key1)
    print("[예제 1] 기본 암호화/복호화")
    print(f"  평문:   {plain1}")
    print(f"  키:     {key1}")
    print(f"  암호문: {cipher1}")
    print(f"  복원:   {decrypt(cipher1, key1)}")
    print()

    # 예제 2: 빈도 분산 시연
    print("[예제 2] 빈도 분산 - 같은 E가 키에 따라 다른 문자로")
    sample = "EEEE"  # E 4개
    for k in ["A", "B", "E", "X"]:
        enc = encrypt(sample, k)
        print(f"  키 {k}: EEEE -> {enc}  (같은 평문, 다른 암호문)")
    print()

    # 예제 3: 반복 구절이 있는 긴 평문 (카시스키 시연용)
    plain3 = "THE cat AND THE dog ARE THE best THE end"
    key3 = "KEY"
    cipher3 = encrypt(plain3, key3)
    print("[예제 3] 카시스키 검사용 - THE가 반복되는 평문")
    print(f"  평문: {plain3}")
    print(f"  키:   {key3}")
    print(f"  암호문: {cipher3}")
    repeats = find_repeated_sequences(cipher3)
    if repeats:
        print("  반복 구절:")
        for seq, pos in list(repeats.items())[:5]:
            dists = [pos[j] - pos[i] for i in range(len(pos)) for j in range(i + 1, len(pos))]
            print(f"    '{seq}' @ {pos} -> 거리들: {dists}")
    kasiski = kasiski_analysis(cipher3)
    print("  키 길이 후보 (카시스키):", [(k, s) for k, s in kasiski[:5]])
    print()

    # 예제 4: 공격 시연 (THE 등 반복이 많은 긴 평문으로 카시스키 효과 극대화)
    plain4 = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGTHEDOGANDTHECATPLAY"
    key4 = "KEY"
    cipher4 = encrypt(plain4, key4)
    print("[예제 4] 배비지 공격 - 키 모른 채 복원")
    print(f"  평문: {plain4[:50]}...")
    print(f"  키:   {key4}")
    print(f"  암호문: {cipher4[:50]}...")
    found_key, found_plain, klen = auto_attack(cipher4)
    print(f"  추정 키: {found_key} (키길이={klen})")
    print(f"  복원:   {found_plain[:50]}...")
    print(f"  일치:   {found_plain == plain4}")
    print()


def interactive_mode() -> None:
    """사용자가 직접 암호화/복호화/공격을 시도하는 대화형 모드."""
    print("\n" + "=" * 60)
    print("대화형 모드: 비즈네르 암호 실습")
    print("  1) 암호화  2) 복호화  3) 공격(키 추정)  4) 종료")
    print("=" * 60)

    while True:
        print("\n선택: 1) 암호화  2) 복호화  3) 공격  4) 종료")
        choice = input("> ").strip()

        if choice == "4":
            print("종료합니다.")
            break

        if choice == "1":
            plain = input("평문 (영문): ").strip()
            key = input("키: ").strip()
            if plain and key:
                cipher = encrypt(plain, key)
                print(f"암호문: {cipher}")

        elif choice == "2":
            cipher = input("암호문: ").strip()
            key = input("키: ").strip()
            if cipher and key:
                plain = decrypt(cipher, key)
                print(f"평문: {plain}")

        elif choice == "3":
            cipher = input("공격할 암호문: ").strip()
            if cipher:
                print("  분석 중...")
                repeats = find_repeated_sequences(cipher)
                kasiski = kasiski_analysis(cipher)
                ic_est = ic_key_length_estimate(cipher)
                print("  [카시스키] 키 길이 후보:", [(k, s) for k, s in kasiski[:5]])
                print("  [일치지수] 키 길이 후보:", [(k, f"{ic:.4f}") for k, ic in ic_est[:5]])
                key, plain, klen = auto_attack(cipher)
                print(f"  추정 키 ({klen}글자): {key}")
                print(f"  복원 평문: {plain}")


def main() -> None:
    """메인 진입점."""
    run_demo()
    ans = input("대화형 모드로 진행하시겠습니까? (y/n): ").strip().lower()
    if ans == "y":
        interactive_mode()


if __name__ == "__main__":
    main()
