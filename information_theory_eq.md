<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    messageStyle: "none",
    tex2jax: { preview: "none" }
  });
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# 정보이론: 자기 정보량과 상호 정보량

## 1. 자기 정보량 (Self-Information)

### 정의
**자기 정보량(Self-Information)** 또는 **정보량(Information Content)** 은 특정 사건이 발생했을 때 얻을 수 있는 정보의 양을 나타냅니다.

> **자기 정보량 공식:**  
> 사건 $x$의 자기 정보량 $I(x)$는 확률 $P(x)$에 대해 다음과 같이 정의됩니다:
> $$I(x) = \log_2 \left( \frac{1}{P(x)} \right) = -\log_2 P(x)$$

### 핵심 개념

1. **확률과 정보량의 관계**
   - 확률이 **낮을수록** (놀라운 사건일수록) 정보량은 **커집니다**
   - 확률이 **높을수록** (예상 가능한 사건일수록) 정보량은 **작습니다**

2. **단위**
   - 로그의 밑이 2일 때: **비트(bit)**
   - 로그의 밑이 $e$일 때: **냇(nat)**
   - 로그의 밑이 10일 때: **하트리(hartley)**

### 예시

**예시 1: 동전 던지기**
- 앞면이 나올 확률: $P(\text{앞면}) = 0.5$
- 자기 정보량: $I(\text{앞면}) = -\log_2(0.5) = 1$ bit

**예시 2: 주사위 던지기**
- 6이 나올 확률: $P(6) = \frac{1}{6}$
- 자기 정보량: $I(6) = -\log_2\left(\frac{1}{6}\right) = \log_2(6) \approx 2.585$ bits

**예시 3: 확실한 사건**
- 확률이 1인 사건: $P(x) = 1$
- 자기 정보량: $I(x) = -\log_2(1) = 0$ bits
- → 예상 가능한 사건이므로 새로운 정보가 없음

---

## 2. 엔트로피 (Entropy)

### 정의
**엔트로피(Entropy)** $H(X)$는 확률변수 $X$의 **불확실성** 또는 **평균 정보량**을 나타냅니다. 클로드 섀넌(Claude Shannon)이 1948년에 정보이론의 기초로 제안한 개념입니다.

> **엔트로피 공식:**  
> 이산 확률변수 $X$가 $n$개의 가능한 값 $\{x_1, x_2, \ldots, x_n\}$을 가지고, 각 값의 확률이 $P(x_i)$일 때:
> $$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i) = \sum_{i=1}^{n} P(x_i) \cdot I(x_i)$$

### 핵심 개념

1. **엔트로피의 의미**
   - 확률변수의 **불확실성**을 측정
   - 모든 사건의 **자기 정보량의 기댓값(평균)**
   - 정보원(source)에서 메시지를 전송할 때 필요한 **평균 비트 수**

2. **엔트로피의 특성**
   - $H(X) \geq 0$: 항상 0 이상
   - $H(X) = 0$: 결과가 **확정적**일 때 (하나의 사건 확률이 1)
   - $H(X)$가 **최대**: 모든 사건이 **균등한 확률**일 때
   - 균등 분포에서 최대 엔트로피: $H_{max} = \log_2 n$

3. **엔트로피와 불확실성**
   - 높은 엔트로피 = 높은 불확실성 = 예측 어려움
   - 낮은 엔트로피 = 낮은 불확실성 = 예측 용이

### 예시

**예시 1: 공정한 동전**
- $P(\text{앞면}) = P(\text{뒷면}) = 0.5$
- $H(X) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1$ bit
- → 최대 불확실성 (균등 분포)

**예시 2: 편향된 동전**
- $P(\text{앞면}) = 0.9$, $P(\text{뒷면}) = 0.1$
- $H(X) = -0.9 \log_2(0.9) - 0.1 \log_2(0.1) \approx 0.469$ bits
- → 결과가 더 예측 가능하므로 엔트로피가 낮음

**예시 3: 확정적 사건**
- $P(\text{앞면}) = 1.0$, $P(\text{뒷면}) = 0$
- $H(X) = -1 \cdot \log_2(1) = 0$ bits
- → 결과가 확정적이므로 불확실성 없음

**예시 4: 공정한 주사위**
- $P(x_i) = \frac{1}{6}$ for $i = 1, 2, \ldots, 6$
- $H(X) = -6 \times \frac{1}{6} \log_2\left(\frac{1}{6}\right) = \log_2(6) \approx 2.585$ bits

---

## 3. 결합 엔트로피 (Joint Entropy)

### 정의
**결합 엔트로피(Joint Entropy)** $H(X, Y)$는 두 확률변수 $X$와 $Y$를 **동시에 관찰**할 때의 불확실성을 나타냅니다.

> **결합 엔트로피 공식:**  
> $$H(X, Y) = -\sum_{x \in X} \sum_{y \in Y} P(x, y) \log_2 P(x, y)$$

여기서 $P(x, y)$는 $X = x$이고 $Y = y$인 **결합 확률(Joint Probability)** 입니다.

### 핵심 개념

1. **결합 엔트로피의 특성**
   - $H(X, Y) \geq 0$: 항상 0 이상
   - $H(X, Y) \geq \max(H(X), H(Y))$: 개별 엔트로피보다 크거나 같음
   - $H(X, Y) \leq H(X) + H(Y)$: 개별 엔트로피 합보다 작거나 같음
   - 등호 성립 조건: $X$와 $Y$가 **독립**일 때

2. **독립일 때의 결합 엔트로피**
   - $X$와 $Y$가 독립이면: $H(X, Y) = H(X) + H(Y)$
   - 두 변수가 독립이면 결합 불확실성 = 개별 불확실성의 합

3. **의존적일 때의 결합 엔트로피**
   - $X$와 $Y$가 의존적이면: $H(X, Y) < H(X) + H(Y)$
   - 한 변수가 다른 변수에 대한 정보를 제공하므로 총 불확실성이 줄어듦

### 예시

**예시 1: 독립적인 두 동전**
- $X$: 첫 번째 동전, $Y$: 두 번째 동전 (독립)
- $H(X) = H(Y) = 1$ bit
- $H(X, Y) = H(X) + H(Y) = 2$ bits

**예시 2: 동일한 동전 (완전 의존)**
- $X = Y$ (같은 결과)
- 결합 확률: $P(\text{앞앞}) = 0.5$, $P(\text{뒷뒷}) = 0.5$, 나머지는 0
- $H(X, Y) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1$ bit
- → $H(X, Y) = H(X) = H(Y)$: 하나를 알면 다른 하나도 알 수 있음

**예시 3: 결합 확률표를 이용한 계산**

| $X$ \ $Y$ | $y_1$ | $y_2$ |
|-----------|-------|-------|
| $x_1$ | 0.4 | 0.1 |
| $x_2$ | 0.1 | 0.4 |

- $H(X, Y) = -(0.4 \log_2 0.4 + 0.1 \log_2 0.1 + 0.1 \log_2 0.1 + 0.4 \log_2 0.4)$
- $H(X, Y) \approx 1.72$ bits

---

## 4. 조건부 엔트로피 (Conditional Entropy)

### 정의
**조건부 엔트로피(Conditional Entropy)** $H(X|Y)$는 확률변수 $Y$를 **이미 알고 있을 때** $X$의 **잔여 불확실성**을 나타냅니다.

> **조건부 엔트로피 공식:**  
> $$H(X|Y) = \sum_{y \in Y} P(y) \cdot H(X|Y=y) = -\sum_{x \in X} \sum_{y \in Y} P(x, y) \log_2 P(x|y)$$

여기서:
- $P(x|y) = \frac{P(x, y)}{P(y)}$: $Y = y$가 주어졌을 때 $X = x$의 **조건부 확률**
- $H(X|Y=y)$: $Y = y$가 주어졌을 때 $X$의 엔트로피

### 핵심 개념

1. **조건부 엔트로피의 의미**
   - $Y$를 관찰한 **후에** $X$에 대해 남아있는 불확실성
   - $Y$가 $X$에 대해 **얼마나 많은 정보를 제공하는지**의 역수 개념

2. **조건부 엔트로피의 특성**
   - $H(X|Y) \geq 0$: 항상 0 이상
   - $H(X|Y) \leq H(X)$: 조건부 엔트로피는 원래 엔트로피보다 작거나 같음
   - $H(X|Y) = 0$: $Y$를 알면 $X$를 **완전히 알 수 있음** (결정적 관계)
   - $H(X|Y) = H(X)$: $Y$가 $X$에 대해 **아무 정보도 제공하지 않음** (독립)

3. **연쇄 법칙 (Chain Rule)**
   - $H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$
   - 결합 엔트로피 = 한 변수의 엔트로피 + 다른 변수의 조건부 엔트로피

### 예시

**예시 1: 독립적인 변수**
- $X$와 $Y$가 독립
- $H(X|Y) = H(X)$
- → $Y$를 알아도 $X$에 대한 불확실성이 줄어들지 않음

**예시 2: 완전히 결정적인 관계**
- $Y = f(X)$ (함수 관계)
- $H(Y|X) = 0$
- → $X$를 알면 $Y$를 완전히 결정할 수 있음

**예시 3: 날씨와 우산**
- $X$: 날씨 (맑음/비), $Y$: 우산 소지 (있음/없음)
- 비가 오면 대부분 우산을 가져간다고 가정
- $H(Y|X) < H(Y)$
- → 날씨를 알면 우산 소지 여부의 불확실성이 줄어듦

**예시 4: 조건부 확률표를 이용한 계산**

결합 확률표:
| $X$ \ $Y$ | $y_1$ | $y_2$ | $P(X)$ |
|-----------|-------|-------|--------|
| $x_1$ | 0.4 | 0.1 | 0.5 |
| $x_2$ | 0.1 | 0.4 | 0.5 |
| $P(Y)$ | 0.5 | 0.5 | 1.0 |

조건부 확률:
- $P(x_1|y_1) = \frac{0.4}{0.5} = 0.8$, $P(x_2|y_1) = 0.2$
- $P(x_1|y_2) = \frac{0.1}{0.5} = 0.2$, $P(x_2|y_2) = 0.8$

조건부 엔트로피:
- $H(X|Y=y_1) = -0.8 \log_2(0.8) - 0.2 \log_2(0.2) \approx 0.722$ bits
- $H(X|Y=y_2) = -0.2 \log_2(0.2) - 0.8 \log_2(0.8) \approx 0.722$ bits
- $H(X|Y) = 0.5 \times 0.722 + 0.5 \times 0.722 \approx 0.722$ bits

검증: $H(X) = 1$ bit이므로, $H(X|Y) < H(X)$ → $Y$가 $X$에 대한 정보를 제공함

---

## 5. 엔트로피 개념 간의 관계

### 관계식 기반 설명

엔트로피 개념들 간의 관계를 수식과 설명으로 표현하면 다음과 같습니다:

#### 1. 기본 관계식

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)$$

#### 2. 개념별 구성 요소

| 개념 | 구성 요소 | 수식 |
|------|----------|------|
| **$H(X)$** | $H(X\|Y) + I(X; Y)$ | $X$의 전체 불확실성 |
| **$H(Y)$** | $H(Y\|X) + I(X; Y)$ | $Y$의 전체 불확실성 |
| **$H(X, Y)$** | $H(X\|Y) + I(X; Y) + H(Y\|X)$ | 두 변수의 결합 불확실성 |
| **$H(X\|Y)$** | $H(X) - I(X; Y)$ | $Y$를 알 때 $X$의 잔여 불확실성 |
| **$H(Y\|X)$** | $H(Y) - I(X; Y)$ | $X$를 알 때 $Y$의 잔여 불확실성 |
| **$I(X; Y)$** | $H(X) - H(X\|Y) = H(Y) - H(Y\|X)$ | 두 변수가 공유하는 정보량 |

#### 3. 시각적 표현 (텍스트 기반)

```
H(X,Y) 전체 영역
├─ H(X|Y) : Y를 알 때 X의 잔여 불확실성
├─ I(X;Y) : X와 Y가 공유하는 정보량
└─ H(Y|X) : X를 알 때 Y의 잔여 불확실성

H(X) = H(X|Y) + I(X;Y)
H(Y) = H(Y|X) + I(X;Y)
H(X,Y) = H(X|Y) + I(X;Y) + H(Y|X)
```

#### 4. 비교표

| 개념 | 기호 | 의미 |
|------|------|------|
| 엔트로피 | $H(X)$ | $X$의 불확실성 (평균 정보량) |
| 결합 엔트로피 | $H(X, Y)$ | $X$와 $Y$를 함께 관찰할 때의 총 불확실성 |
| 조건부 엔트로피 | $H(X\|Y)$ | $Y$를 알 때 $X$의 잔여 불확실성 |
| 상호 정보량 | $I(X; Y)$ | $X$와 $Y$가 공유하는 정보량 |

---

## 6. 상호 정보량 (Mutual Information)

### 정의
**상호 정보량(Mutual Information)** $I(X; Y)$는 두 확률변수 $X$와 $Y$ 사이의 **상호 의존성**을 측정합니다. 즉, 한 확률변수를 관찰함으로써 다른 확률변수에 대해 얻을 수 있는 정보의 양을 나타냅니다.

> **상호 정보량 공식:**  
> $$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} P(x, y) \log_2 \left( \frac{P(x, y)}{P(x) P(y)} \right)$$

또는 엔트로피를 이용하여 다음과 같이 표현할 수 있습니다:
$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)$$

여기서:
- $H(X)$: $X$의 엔트로피
- $H(Y)$: $Y$의 엔트로피
- $H(X|Y)$: $Y$가 주어졌을 때 $X$의 조건부 엔트로피
- $H(X, Y)$: $X$와 $Y$의 결합 엔트로피

### 핵심 개념

1. **대칭성**
   - $I(X; Y) = I(Y; X)$
   - $X$가 $Y$에 대해 제공하는 정보 = $Y$가 $X$에 대해 제공하는 정보

2. **범위**
   - $I(X; Y) \geq 0$ (항상 0 이상)
   - $I(X; Y) = 0$: $X$와 $Y$가 **독립**일 때
   - $I(X; Y)$가 클수록: $X$와 $Y$ 사이의 **의존성이 강함**

3. **관계**
   - $I(X; Y) \leq \min(H(X), H(Y))$
   - 상호 정보량은 각 변수의 엔트로피보다 클 수 없음

### 예시

**예시 1: 완전히 독립적인 변수**
- $X$와 $Y$가 독립: $P(x, y) = P(x) \cdot P(y)$
- 상호 정보량: $I(X; Y) = 0$ bits
- → 한 변수를 알아도 다른 변수에 대한 정보를 얻을 수 없음

**예시 2: 완전히 의존적인 변수**
- $X = Y$ (완전히 동일한 변수)
- 상호 정보량: $I(X; Y) = H(X) = H(Y)$
- → 한 변수를 알면 다른 변수를 완전히 알 수 있음

**예시 3: 날씨와 우산 판매**
- $X$: 날씨 (맑음/비)
- $Y$: 우산 판매량 (높음/낮음)
- 비가 올 때 우산 판매량이 높다면 $I(X; Y) > 0$
- → 날씨를 알면 우산 판매량에 대한 정보를 얻을 수 있음

---

## 7. 자기 정보량 vs 상호 정보량 비교

| 특성 | 자기 정보량 $I(x)$ | 상호 정보량 $I(X; Y)$ |
|------|-------------------|---------------------|
| **대상** | 단일 사건 $x$ | 두 확률변수 $X$, $Y$ |
| **의미** | 특정 사건의 정보량 | 두 변수 간 공유 정보량 |
| **범위** | $I(x) \geq 0$ | $I(X; Y) \geq 0$ |
| **독립성** | - | 독립일 때 $I(X; Y) = 0$ |
| **관계** | 엔트로피의 구성 요소 | 엔트로피의 차이로 표현 가능 |

---

## 8. Python 실습 코드

```python
import math
import numpy as np

def self_information(probability):
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

def mutual_information(joint_prob, x_prob, y_prob):
    """
    상호 정보량을 계산하는 함수
    
    Args:
        joint_prob: 결합 확률 P(X, Y) (2D 배열)
        x_prob: X의 주변 확률 P(X) (1D 배열)
        y_prob: Y의 주변 확률 P(Y) (1D 배열)
    
    Returns:
        상호 정보량 (bits)
    """
    mi = 0.0
    for i in range(len(x_prob)):
        for j in range(len(y_prob)):
            if joint_prob[i][j] > 0:
                mi += joint_prob[i][j] * math.log2(
                    joint_prob[i][j] / (x_prob[i] * y_prob[j])
                )
    return mi

def entropy(probabilities):
    """
    엔트로피를 계산하는 함수
    
    Args:
        probabilities: 확률 분포 (리스트)
    
    Returns:
        엔트로피 (bits)
    """
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def joint_entropy(joint_prob):
    """
    결합 엔트로피를 계산하는 함수
    
    Args:
        joint_prob: 결합 확률 P(X, Y) (2D 배열)
    
    Returns:
        결합 엔트로피 (bits)
    """
    return entropy(joint_prob.flatten())

def conditional_entropy(joint_prob, given_prob):
    """
    조건부 엔트로피 H(X|Y)를 계산하는 함수
    
    Args:
        joint_prob: 결합 확률 P(X, Y) (2D 배열, 행=X, 열=Y)
        given_prob: 조건 변수 Y의 주변 확률 P(Y) (1D 배열)
    
    Returns:
        조건부 엔트로피 H(X|Y) (bits)
    """
    h_cond = 0.0
    for j in range(len(given_prob)):
        if given_prob[j] > 0:
            cond_prob = joint_prob[:, j] / given_prob[j]
            h_given_y = entropy(cond_prob)
            h_cond += given_prob[j] * h_given_y
    return h_cond

# 예시 1: 자기 정보량 계산
print("=== 자기 정보량 예시 ===")
print(f"동전 앞면 (P=0.5): {self_information(0.5):.3f} bits")
print(f"주사위 6 (P=1/6): {self_information(1/6):.3f} bits")
print(f"확실한 사건 (P=1.0): {self_information(1.0):.3f} bits")
print(f"드문 사건 (P=0.01): {self_information(0.01):.3f} bits")
print()

# 예시 2: 상호 정보량 계산
print("=== 상호 정보량 예시 ===")
joint_indep = np.array([[0.25, 0.25], [0.25, 0.25]])
x_prob_indep = np.array([0.5, 0.5])
y_prob_indep = np.array([0.5, 0.5])
mi_indep = mutual_information(joint_indep, x_prob_indep, y_prob_indep)
print(f"독립적인 변수: I(X;Y) = {mi_indep:.3f} bits")

joint_dep = np.array([[0.5, 0.0], [0.0, 0.5]])
x_prob_dep = np.array([0.5, 0.5])
y_prob_dep = np.array([0.5, 0.5])
mi_dep = mutual_information(joint_dep, x_prob_dep, y_prob_dep)
print(f"완전 의존 변수: I(X;Y) = {mi_dep:.3f} bits")

joint_partial = np.array([[0.4, 0.1], [0.1, 0.4]])
x_prob_partial = np.array([0.5, 0.5])
y_prob_partial = np.array([0.5, 0.5])
mi_partial = mutual_information(joint_partial, x_prob_partial, y_prob_partial)
print(f"부분 의존 변수: I(X;Y) = {mi_partial:.3f} bits")
print()

# 예시 3: 엔트로피, 결합/조건부 엔트로피
print("=== 엔트로피 예시 ===")
fair_coin = [0.5, 0.5]
print(f"공정한 동전: H = {entropy(fair_coin):.3f} bits")
biased_coin = [0.9, 0.1]
print(f"편향된 동전 (0.9/0.1): H = {entropy(biased_coin):.3f} bits")
fair_dice = [1/6] * 6
print(f"공정한 주사위: H = {entropy(fair_dice):.3f} bits")
print()

print("=== 결합 엔트로피와 조건부 엔트로피 ===")
joint_example = np.array([[0.4, 0.1], [0.1, 0.4]])
x_prob_example = np.array([0.5, 0.5])
y_prob_example = np.array([0.5, 0.5])
h_x_ex = entropy(x_prob_example)
h_y_ex = entropy(y_prob_example)
h_xy_ex = joint_entropy(joint_example)
h_x_given_y = conditional_entropy(joint_example, y_prob_example)
h_y_given_x = conditional_entropy(joint_example.T, x_prob_example)
print(f"H(X) = {h_x_ex:.3f} bits, H(Y) = {h_y_ex:.3f} bits")
print(f"H(X,Y) = {h_xy_ex:.3f} bits")
print(f"H(X|Y) = {h_x_given_y:.3f} bits, H(Y|X) = {h_y_given_x:.3f} bits")
print()
print("=== 관계식 검증 ===")
print(f"I(X;Y) = H(X) - H(X|Y) = {h_x_ex - h_x_given_y:.3f} bits")
print(f"I(X;Y) = H(X) + H(Y) - H(X,Y) = {h_x_ex + h_y_ex - h_xy_ex:.3f} bits")
print(f"연쇄 법칙: H(X,Y) = H(X) + H(Y|X) = {h_x_ex + h_y_given_x:.3f} bits")
```

---

## 9. 요약

### 자기 정보량 $I(x)$
- **정의**: $I(x) = -\log_2 P(x)$
- **의미**: 특정 사건이 발생했을 때 얻는 정보의 양
- **특징**: 확률이 낮을수록 정보량이 큼

### 엔트로피 $H(X)$
- **정의**: $H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$
- **의미**: 확률변수의 불확실성 (자기 정보량의 평균)
- **특징**: 균등 분포에서 최대, 확정적일 때 0

### 결합 엔트로피 $H(X, Y)$
- **정의**: $H(X, Y) = -\sum_{x} \sum_{y} P(x, y) \log_2 P(x, y)$
- **의미**: 두 변수를 동시에 관찰할 때의 총 불확실성
- **특징**: $H(X, Y) \leq H(X) + H(Y)$ (등호: 독립일 때)

### 조건부 엔트로피 $H(X|Y)$
- **정의**: $H(X|Y) = -\sum_{x} \sum_{y} P(x, y) \log_2 P(x|y)$
- **의미**: $Y$를 알 때 $X$의 잔여 불확실성
- **특징**: $H(X|Y) \leq H(X)$, 연쇄 법칙: $H(X, Y) = H(X) + H(Y|X)$

### 상호 정보량 $I(X; Y)$
- **정의**: $I(X; Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X, Y)$
- **의미**: 두 변수 간 공유되는 정보의 양
- **특징**: 독립일 때 0, 의존성이 강할수록 큰 값, 대칭성 $I(X; Y) = I(Y; X)$

### 핵심 관계식
$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$
$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)$$

### 실용적 활용
- **자기 정보량**: 데이터 압축, 부호화 이론
- **엔트로피**: 데이터 압축 한계, 언어 모델 평가 (perplexity)
- **결합/조건부 엔트로피**: 정보 흐름 분석, 인과관계 추론
- **상호 정보량**: 특징 선택, 변수 간 관계 분석, 정보 이론적 머신러닝
