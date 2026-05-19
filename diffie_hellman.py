"""
Diffie-Hellman 키 교환 (Key Exchange) 실습 프로그램

Whitfield Diffie & Martin Hellman (1976)의 키 분배 문제 해결 방안을 구현합니다.
- 공개 파라미터: 소수 p, 생성자 g
- Alice와 Bob이 각각 비밀값 a, b를 가지고 공유 비밀 K = g^(ab) mod p 를 생성
- 도청자는 g, p, A, B만 알 수 있어 이산 로그 문제로 인해 K를 구하기 어렵다

Diffie_Hellman_키분배_설명자료.md와 함께 사용하세요.
"""

import random
from typing import Tuple


# ---------------------------------------------------------------------------
# 소수 및 생성자 (실습용: 작은 수 사용. 실제로는 수백~수천 비트 사용)
# ---------------------------------------------------------------------------

# 표준 모드용 중간 크기 소수 (원시근 탐색이 가능한 크기)
# 실제 표준에서는 2048비트 이상 권장 (RFC 3526 등)
DEMO_PRIME = 2**19 - 1  # 메르센 소수 524287 (실습용)

# 위 소수가 너무 크면 실습에서 지수 연산이 느릴 수 있으므로, 작은 실습용도 제공
SMALL_PRIME = 23   # 작은 소수 (실습·시연용)
SMALL_GENERATOR = 5  # 23에 대한 원시근(생성자) 예: 5^1,5^2,... mod 23 이 1~22를 생성


def is_prime(n: int) -> bool:
    """n이 소수인지 간단히 판별 (작은 n용)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for d in range(3, int(n ** 0.5) + 1, 2):
        if n % d == 0:
            return False
    return True


def find_primitive_root(p: int) -> int:
    """
    소수 p에 대한 원시근(생성자) 하나를 찾습니다.
    g가 원시근이면 g^1, g^2, ..., g^(p-1) mod p 가 서로 다르고 1..p-1을 모두 생성.
    """
    if not is_prime(p):
        raise ValueError("p must be prime")
    # p-1의 소인수들 (간단히 2와 (p-1)//2 정도만 확인)
    phi = p - 1
    for g in range(2, p):
        ok = True
        x = phi
        # g^phi mod p = 1 이고, 더 작은 지수에서 1이 되면 안 됨
        cur = pow(g, phi, p)
        if cur != 1:
            continue
        # phi의 진약수 d에 대해 g^d mod p != 1 확인
        for d in [2, phi // 2] if phi % 2 == 0 else [phi // 2]:
            if d <= 0 or d >= phi:
                continue
            if pow(g, d, p) == 1:
                ok = False
                break
        if ok:
            return g
    return 2  # fallback


def mod_pow(base: int, exp: int, mod: int) -> int:
    """base^exp mod mod 를 빠르게 계산 (내장 pow 사용)."""
    return pow(base, exp, mod)


# ---------------------------------------------------------------------------
# Diffie-Hellman 키 교환
# ---------------------------------------------------------------------------

def dh_key_pair(p: int, g: int) -> Tuple[int, int]:
    """
    한 당사자의 비밀값과 공개값을 생성합니다.
    Returns: (private_key, public_key)  즉 (a, A = g^a mod p)
    """
    # 비밀값: 2 ~ p-2 사이의 난수 (1이나 p-1은 보안상 약함)
    private = random.randrange(2, p - 1)
    public = mod_pow(g, private, p)
    return private, public


def dh_shared_secret(private_key: int, other_public: int, p: int) -> int:
    """
    자신의 비밀키와 상대방의 공개값으로 공유 비밀을 계산합니다.
    K = other_public^private_key mod p = g^(ab) mod p
    """
    return mod_pow(other_public, private_key, p)


# ---------------------------------------------------------------------------
# 시연: Alice와 Bob의 키 교환
# ---------------------------------------------------------------------------

def run_demo(use_small_params: bool = True) -> None:
    """
    Diffie-Hellman 키 교환을 Alice와 Bob으로 시연합니다.
    use_small_params=True 이면 작은 p,g로 빠르게, False면 큰 소수 사용.
    """
    if use_small_params:
        p, g = SMALL_PRIME, SMALL_GENERATOR
        print("[실습 모드] 작은 소수와 생성자 사용 (p=23, g=5)")
    else:
        p = DEMO_PRIME
        g = find_primitive_root(p)
        print("[표준 모드] 큰 소수 p 사용 (일부만 표시)")

    print(f"  p = {p if use_small_params else str(p)[:50] + '...'}")
    print(f"  g = {g}")
    print()

    # Alice: 비밀 a, 공개 A = g^a mod p
    a, A = dh_key_pair(p, g)
    print("Alice: 비밀 a 선택, A = g^a mod p 계산 후 Bob에게 A 전송")
    print(f"  (비밀 a = {a}, 공개 A = {A})")
    print()

    # Bob: 비밀 b, 공개 B = g^b mod p
    b, B = dh_key_pair(p, g)
    print("Bob: 비밀 b 선택, B = g^b mod p 계산 후 Alice에게 B 전송")
    print(f"  (비밀 b = {b}, 공개 B = {B})")
    print()

    # 공유 비밀
    K_alice = dh_shared_secret(a, B, p)
    K_bob = dh_shared_secret(b, A, p)

    print("공유 비밀 계산:")
    print(f"  Alice: K = B^a mod p = {K_alice}")
    print(f"  Bob:   K = A^b mod p = {K_bob}")
    print(f"  일치 여부: {K_alice == K_bob}")
    print()

    # 도청자 관점
    print("도청자(Eve)가 볼 수 있는 것: p, g, A, B")
    print("  → K = g^(ab) mod p 를 얻으려면 a 또는 b(이산 로그)를 알아야 함 → 어려움")
    return K_alice


# ---------------------------------------------------------------------------
# 공유 비밀을 이용한 간단 대칭키 시연 (선택)
# ---------------------------------------------------------------------------

def demo_shared_key_encryption(shared_key: int) -> None:
    """
    공유 비밀을 정수 키로 사용해 간단히 XOR 스타일로
    같은 키를 쓰면 같은 결과가 나옴을 보여줍니다.
    (실제 암호화는 AES 등 표준 알고리즘 사용)
    """
    msg = "HELLO"
    # 키를 바이트 스트림처럼 쓰기 위해 해시처럼 단순 변환 (실습용)
    key_bytes = (shared_key % (256 ** 4)).to_bytes(4, "big")
    key_stream = (key_bytes * (len(msg) + 1))[: len(msg)]

    encoded = bytes(ord(msg[i]) ^ key_stream[i] for i in range(len(msg)))
    decoded = bytes(encoded[i] ^ key_stream[i] for i in range(len(msg)))

    print("공유 비밀을 이용한 간단 XOR 시연 (실습용):")
    print(f"  원문: {msg}")
    print(f"  인코딩: {encoded.hex()} (hex)")
    print(f"  디코딩: {decoded.decode('ascii')}")
    print()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Diffie-Hellman 키 교환 실습")
    print("=" * 60)
    print()

    K = run_demo(use_small_params=True)
    print("-" * 60)
    demo_shared_key_encryption(K)

    print("실습을 반복하려면 run_demo(use_small_params=True)를 다시 호출하세요.")
    print("매번 다른 a, b로 다른 공유 비밀이 생성됩니다.")
