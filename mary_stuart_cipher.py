"""
Mary Queen of Scots 암호 실습 프로그램
- Babington 스타일 단일 치환 암호 + Null 문자 + 반복 부호 + 명칭자(단어 코드)
- 빈도 분석 시연
"""

import random
import string
from collections import Counter
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 암호 상수 (Babington 암호 스타일)
# ---------------------------------------------------------------------------

# 평문 알파벳 (16세기 영어: J와 I, U와 V가 혼용됨 → 26자 사용)
PLAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 치환 암호 알파벳 (섞인 순서 - 키 역할)
# 역사적으로는 수신자와 발신자만 알고 있는 비밀 표였음
DEFAULT_CIPHER_ALPHABET = "QWERTYUIOPASDFGHJKLZXCVBNM"

# Null 문자 5개 (의미 없음, 빈도 분석 방해) - ASCII 호환
NULL_SYMBOLS = ["@", "#", "$", "%", "&"]

# 반복 부호 (다음 심볼이 두 번 반복됨을 나타냄)
DOUBLE_SYMBOL = "="

# 명칭자: 자주 쓰는 단어 -> 한 글자 코드 (선택 사용) - ASCII 호환
# 0-9는 치환 알파벳(A-Z)과 겹치지 않음
NOMENCLATOR = {
    "QUEEN": "0",
    "KING": "1",
    "PRINCE": "2",
    "MARY": "3",
    "ELIZABETH": "4",
    "LETTER": "5",
    "FRIEND": "6",
    "LORD": "7",
    "ENGLAND": "8",
    "SCOTLAND": "9",
}


# ---------------------------------------------------------------------------
# 치환 테이블 구성
# ---------------------------------------------------------------------------

def build_substitution_table(
    cipher_alphabet: str = DEFAULT_CIPHER_ALPHABET,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    평문 → 암호문, 암호문 → 평문 치환 테이블을 구성합니다.
    """
    if len(cipher_alphabet) != 26 or len(set(cipher_alphabet)) != 26:
        raise ValueError("암호 알파벳은 26개의 서로 다른 문자여야 합니다.")
    enc = {p: c for p, c in zip(PLAIN_ALPHABET, cipher_alphabet.upper())}
    dec = {c: p for p, c in enc.items()}
    return enc, dec


# ---------------------------------------------------------------------------
# 인코딩 (암호화)
# ---------------------------------------------------------------------------

def encode_substitution(
    text: str,
    enc_table: Dict[str, str],
    use_nulls: bool = True,
    use_double: bool = True,
    null_prob: float = 0.15,
) -> str:
    """
    평문을 Babington 스타일 암호로 인코딩합니다.
    - use_nulls: Null 문자를 무작위로 삽입 (빈도 분석 방해)
    - use_double: 연속된 동일 문자를 반복 부호로 압축
    - null_prob: Null 삽입 확률 (0~1)
    """
    text = text.upper().strip()
    result: List[str] = []

    i = 0
    while i < len(text):
        c = text[i]

        # 명칭자: 등록된 단어가 있으면 해당 심볼로 치환
        found_word = False
        for word, sym in NOMENCLATOR.items():
            if text[i : i + len(word)] == word:
                result.append(sym)
                i += len(word)
                found_word = True
                break
        if found_word:
            if use_nulls and random.random() < null_prob:
                result.append(random.choice(NULL_SYMBOLS))
            continue

        # 일반 문자
        if c in enc_table:
            # 연속 문자: AA → ‡A
            if use_double and i + 1 < len(text) and text[i + 1] == c:
                result.append(DOUBLE_SYMBOL)
                result.append(enc_table[c])
                i += 2
            else:
                result.append(enc_table[c])
                i += 1
        elif c in " \t\n":
            result.append(c)
            i += 1
        else:
            i += 1  # 알 수 없는 문자는 건너뜀

        if use_nulls and c in PLAIN_ALPHABET and random.random() < null_prob:
            result.append(random.choice(NULL_SYMBOLS))

    return "".join(result)


# ---------------------------------------------------------------------------
# 디코딩 (복호화)
# ---------------------------------------------------------------------------

def decode_substitution(
    ciphertext: str,
    dec_table: Dict[str, str],
    nomenclator_rev: Dict[str, str],
) -> str:
    """
    암호문을 평문으로 복호화합니다.
    """
    result: List[str] = []
    i = 0

    while i < len(ciphertext):
        c = ciphertext[i]

        # Null 문자: 무시
        if c in NULL_SYMBOLS:
            i += 1
            continue

        # 반복 부호: 다음 문자를 두 번 출력
        if c == DOUBLE_SYMBOL:
            i += 1
            if i < len(ciphertext):
                next_c = ciphertext[i]
                if next_c in dec_table:
                    result.append(dec_table[next_c] * 2)
                i += 1
            continue

        # 명칭자 (단어 코드)
        if c in nomenclator_rev:
            result.append(nomenclator_rev[c])
            i += 1
            continue

        # 일반 치환
        if c in dec_table:
            result.append(dec_table[c])
        elif c in " \t\n":
            result.append(c)
        i += 1

    return "".join(result)


# ---------------------------------------------------------------------------
# 빈도 분석
# ---------------------------------------------------------------------------

def frequency_analysis(ciphertext: str) -> List[Tuple[str, int, float]]:
    """
    암호문에서 각 심볼의 빈도와 비율을 계산합니다.
    빈도 분석 공격의 기초가 됩니다.
    """
    # Null과 공백 제외
    relevant = [c for c in ciphertext if c not in NULL_SYMBOLS and c != " " and c != "\t" and c != "\n"]
    total = len(relevant)
    if total == 0:
        return []

    counts = Counter(relevant)
    # 빈도 내림차순
    sorted_items = counts.most_common()
    return [(sym, cnt, cnt / total * 100) for sym, cnt in sorted_items]


def print_frequency_table(freq_list: List[Tuple[str, int, float]]) -> None:
    """빈도 분석 결과를 표로 출력합니다."""
    print("  심볼 | 출현 횟수 | 비율(%)")
    print("  " + "-" * 30)
    for sym, cnt, pct in freq_list[:20]:  # 상위 20개
        bar = "*" * min(20, int(pct)) + "-" * (20 - min(20, int(pct)))
        print(f"  {sym!r:^4} | {cnt:8} | {pct:5.2f}% {bar}")
    if len(freq_list) > 20:
        print(f"  ... (총 {len(freq_list)}개 서로 다른 심볼)")


# ---------------------------------------------------------------------------
# 빈도 기반 추측 (교육용 단순 버전)
# ---------------------------------------------------------------------------

# 영어 글자 빈도 (대략적 순서)
ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def suggest_mapping_from_frequency(
    freq_list: List[Tuple[str, int, float]],
) -> Dict[str, str]:
    """
    빈도 분석 결과를 바탕으로 암호 심볼 → 평문 추정 매핑을 제안합니다.
    (교육용: 완벽한 해독이 아닌 '추측' 시연)
    """
    mapping: Dict[str, str] = {}
    for rank, (sym, _, _) in enumerate(freq_list):
        if rank < len(ENGLISH_FREQ_ORDER):
            mapping[sym] = ENGLISH_FREQ_ORDER[rank]
    return mapping


# ---------------------------------------------------------------------------
# 대화형 메뉴 및 예제
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """설명과 함께 기본 예제를 실행합니다."""
    print("=" * 60)
    print("Mary Queen of Scots 암호 실습")
    print("=" * 60)
    print("\n[역사적 배경]")
    print("  Mary Stuart(1542-1587)는 Babington 음모(1586)에서")
    print("  단일 치환 암호 + Null + 명칭자를 사용했습니다.")
    print("  영국 첩보관 Thomas Phelippes에 의해 해독되어")
    print("  메리는 1587년 참수형에 처해졌습니다.")
    print()

    enc_table, dec_table = build_substitution_table()
    nomenclator_rev = {v: k for k, v in NOMENCLATOR.items()}

    # 예제 메시지
    plain = "MARY QUEEN OF SCOTS"
    print("[예제 1] 단순 치환 (Null 없음)")
    cipher1 = encode_substitution(plain, enc_table, use_nulls=False, use_double=False)
    print(f"  평문:   {plain}")
    print(f"  암호문: {cipher1}")
    print(f"  복원:   {decode_substitution(cipher1, dec_table, nomenclator_rev)}")
    print()

    print("[예제 2] Null 문자 삽입 (빈도 분석 방해)")
    random.seed(42)
    cipher2 = encode_substitution(plain, enc_table, use_nulls=True, null_prob=0.2)
    print(f"  평문:   {plain}")
    print(f"  암호문: {cipher2}")
    print(f"  복원:   {decode_substitution(cipher2, dec_table, nomenclator_rev)}")
    print()

    print("[예제 3] 반복 부호 (e.g., BETTER -> B+ =E +TT+ =E +R)")
    plain3 = "BETTER"
    cipher3 = encode_substitution(plain3, enc_table, use_nulls=False, use_double=True)
    print(f"  평문:   {plain3}")
    print(f"  암호문: {cipher3}")
    print(f"  복원:   {decode_substitution(cipher3, dec_table, nomenclator_rev)}")
    print()

    print("[예제 4] 명칭자 (QUEEN, KING 등 단어 코드)")
    plain4 = "THE QUEEN OF ENGLAND"
    cipher4 = encode_substitution(plain4, enc_table, use_nulls=False)
    print(f"  평문:   {plain4}")
    print(f"  암호문: {cipher4}")
    print(f"  복원:   {decode_substitution(cipher4, dec_table, nomenclator_rev)}")
    print()

    print("[예제 5] 빈도 분석 시연")
    sample = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" * 5
    sample += " E" * 20 + " T" * 15 + " A" * 12  # E, T, A 강조
    cipher5 = encode_substitution(sample, enc_table, use_nulls=False)
    freq = frequency_analysis(cipher5)
    print("  (영문 샘플 암호화 후 심볼 빈도)")
    print_frequency_table(freq)
    print()
    print("  ※ 영어에서 E, T, A가 가장 자주 나오므로,")
    print("    암호문에서 가장 빈번한 심볼이 이들에 해당할 가능성이 높습니다.")
    print()


def interactive_mode() -> None:
    """사용자가 직접 암호화/복호화를 시도하는 대화형 모드입니다."""
    enc_table, dec_table = build_substitution_table()
    nomenclator_rev = {v: k for k, v in NOMENCLATOR.items()}

    print("\n" + "=" * 60)
    print("대화형 모드: 직접 암호화/복호화를 시도해 보세요.")
    print("  [치환표] A→Q, B→W, C→E, ... (키: QWERTYUIOPASDFGHJKLZXCVBNM)")
    print("  [명칭자] QUEEN→♔, KING→♕, MARY→♗, ...")
    print("  종료: quit")
    print("=" * 60)

    while True:
        print("\n1) 암호화  2) 복호화  3) 빈도분석  4) 종료")
        choice = input("선택: ").strip()

        if choice == "4" or choice.lower() == "quit":
            print("종료합니다.")
            break

        if choice == "1":
            text = input("평문 입력 (영문): ").strip()
            if not text:
                continue
            use_null = input("Null 문자 삽입? (y/n, 기본 n): ").strip().lower() != "n"
            p = 0.15 if use_null else 0.0
            cipher = encode_substitution(text, enc_table, use_nulls=use_null, null_prob=p)
            print(f"암호문: {cipher}")

        elif choice == "2":
            text = input("암호문 입력: ").strip()
            if not text:
                continue
            dec = decode_substitution(text, dec_table, nomenclator_rev)
            print(f"평문: {dec}")

        elif choice == "3":
            text = input("분석할 암호문 입력: ").strip()
            if not text:
                continue
            freq = frequency_analysis(text)
            print_frequency_table(freq)


def main() -> None:
    """메인 진입점."""
    run_demo()

    ans = input("대화형 모드로 진행하시겠습니까? (y/n): ").strip().lower()
    if ans == "y":
        interactive_mode()


if __name__ == "__main__":
    main()
