# 样本 7: q-448 / qutianyi / sample 0

## 定位信息

- annotation_uid: `q-448__qutianyi__dev-1776932087401-vma9h3__sample_0`
- 原始标注 detail_path: `annotation_app/data/annotations/qutianyi/dev-1776932087401-vma9h3/q-448.detail.json`
- official_path: `official_annotation_data/part_04.jsonl`
- clean input slice line: 7
- remote gen_only source line: 139

## 全链路计数

- reference steps: 6
- reference claims: 68
- generate.py actual prompts: 6
- Kimi/Moonshot prompt-pack prompts: 5
- model gen_output entries: 6
- judge cache rows: 38
- judge route counts: `{'pairwise': 3, 'holistic': 4, 'selfjudge_without_reference': 4, 'selfjudge_with_reference': 27}`
- local repack row-count match: `True`

## 题目

Consider all polynomials $P(x)$ with real coefficients that have the following property: for any two real numbers $x$ and $y$ one has \[|y^2-P(x)|\le 2|x|\quad\text{if and only if}\quad |x^2-P(y)|\le 2|y|.\] Determine all possible values of $P(0)$.

[i]

## 参考答案 / 标准解

{P(0) \in (-\infty,0)\cup \{1\} }

## 被保留的正确 sample 解答

To determine all possible values of \(P(0)\), write the given condition as

\[
|y^2-P(x)|\le 2|x|
\quad\Longleftrightarrow\quad
|x^2-P(y)|\le 2|y|
\qquad (\forall x,y\in \mathbb R).
\]

We will extract information about \(P(0)\) by choosing special values of \(x\) and \(y\).

### Step 1: Set \(x=0\)

If \(x=0\), then the left-hand inequality becomes

\[
|y^2-P(0)|\le 0,
\]

which is equivalent to

\[
y^2=P(0).
\]

Hence, for every real \(y\),

\[
y^2=P(0)
\quad\Longleftrightarrow\quad
|0-P(y)|\le 2|y|,
\]

that is,

\[
y^2=P(0)
\quad\Longleftrightarrow\quad
|P(y)|\le 2|y|.
\]

So the set of real \(y\) satisfying \(|P(y)|\le 2|y|\) is exactly the set of real solutions of \(y^2=P(0)\).

---

### Step 2: Interpret this according to the sign of \(P(0)\)

There are three cases.

#### Case 1: \(P(0)<0\)

Then the equation \(y^2=P(0)\) has no real solution. Therefore the condition from Step 1 says

\[
|P(y)|\le 2|y|
\]

has no real solution \(y\).

In particular, at \(y=0\) we must have

\[
|P(0)|\le 0
\]

false, which is indeed true since \(P(0)<0\). So negative values are at least not immediately excluded.

#### Case 2: \(P(0)=0\)

Then \(y^2=P(0)\) has the unique solution \(y=0\). Thus Step 1 implies

\[
|P(y)|\le 2|y|
\quad\text{holds only at }y=0.
\]

But a polynomial with \(P(0)=0\) is divisible by \(y\): write

\[
P(y)=yQ(y)
\]

for some polynomial \(Q\). Then for \(y\) sufficiently close to \(0\),

\[
|P(y)|=|y||Q(y)|.
\]

Since \(Q\) is continuous, \(|Q(y)|\) is bounded near \(0\), so for all sufficiently small \(y\),

\[
|P(y)|\le 2|y|.
\]

Thus the inequality would hold for infinitely many \(y\) near \(0\), not only for \(y=0\). Contradiction.

Hence \(P(0)\ne 0\).

#### Case 3: \(P(0)>0\)

Then \(y^2=P(0)\) has exactly two real solutions \(y=\pm \sqrt{P(0)}\). So Step 1 gives

\[
|P(y)|\le 2|y|
\quad\text{if and only if}\quad
y=\pm \sqrt{P(0)}.
\]

That means the polynomial

\[
P(y)^2-4y^2
\]

is \(\le 0\) exactly at those two points and \(>0\) elsewhere. Since this is a polynomial, each of those two points must be roots of even multiplicity unless the sign changes. But here it must be nonpositive only at isolated points, so both must actually be double roots and no others. Hence

\[
P(y)^2-4y^2 = c\,(y^2-P(0))^2
\]

for some constant \(c>0\).

Now compare degrees. The left-hand side is \(P(y)^2-4y^2\). For this to be a constant multiple of a quadratic squared, it must have degree \(4\), \(2\), or \(0\). This forces \(\deg P\le 2\). Let us instead use the values at the special points \(y=\pm \sqrt{P(0)}\): there,

\[
|P(y)|=2|y|=2\sqrt{P(0)}.
\]

So for \(a=\sqrt{P(0)}\),

\[
P(a)=\pm 2a,\qquad P(-a)=\pm 2a.
\]

Also, because \(P(y)^2-4y^2\) vanishes only at \(y=\pm a\), the factorization must be exact:

\[
P(y)^2-4y^2 = c(y^2-a^2)^2.
\]

Set \(y=0\). Then

\[
P(0)^2=c a^4.
\]

Since \(a^2=P(0)\), this gives \(P(0)^2=cP(0)^2\), so \(c=1\). Therefore

\[
P(y)^2-4y^2=(y^2-P(0))^2.
\]

Hence

\[
P(y)^2 = y^4 +(4-2P(0))y^2 + P(0)^2.
\]

Now set \(y=0\): this is consistent. To make the right side a perfect square polynomial, we must have

\[
y^4 +(4-2P(0))y^2 + P(0)^2 = (y^2+b)^2
\]

for some real \(b\), because there is no odd-power term. Expanding:

\[
(y^2+b)^2 = y^4+2by^2+b^2.
\]

Thus

\[
2b=4-2P(0),\qquad b^2=P(0)^2.
\]

So \(b=\pm P(0)\). If \(b=P(0)\), then

\[
2P(0)=4-2P(0)\quad\Rightarrow\quad 4P(0)=4\quad\Rightarrow\quad P(0)=1.
\]

If \(b=-P(0)\), then

\[
-2P(0)=4-2P(0)\quad\Rightarrow\quad 0=4,
\]

impossible. Therefore the only positive possibility is

\[
P(0)=1.
\]

---

### Step 3: Collect all possibilities

From the three cases:

- \(P(0)<0\) is possible,
- \(P(0)=0\) is impossible,
- \(P(0)>0\) forces \(P(0)=1\).

So the only possible values are

\[
P(0)\in (-\infty,0)\cup\{1\}.
\]

## 标注者决策摘要

特殊值代入与平方差重根法

## 标注 steps 与 claims

### s1

To determine all possible values of \(P(0)\), write the given condition as

\[
|y^2-P(x)|\le 2|x|
\quad\Longleftrightarrow\quad
|x^2-P(y)|\le 2|y|
\qquad (\forall x,y\in \mathbb R).
\]

We will extract information about \(P(0)\) by choosing special values of \(x\) and \(y\).

Claims:

- `s1c1` 标注判断=correct: To determine all possible values of \(P(0)\), write the given condition as 
\[
|y^2-P(x)|\le 2|x|
\quad\Longleftrightarrow\quad
|x^2-P(y)|\le 2|y|
\qquad (\forall x,y\in \mathbb R).
\]
- `s1c2` 标注判断=correct: We will extract information about \(P(0)\) by choosing special values of \(x\) and \(y\).

### s2

### Step 1: Set \(x=0\)

If \(x=0\), then the left-hand inequality becomes

\[
|y^2-P(0)|\le 0,
\]

which is equivalent to

\[
y^2=P(0).
\]

Hence, for every real \(y\),

\[
y^2=P(0)
\quad\Longleftrightarrow\quad
|0-P(y)|\le 2|y|,
\]

that is,

\[
y^2=P(0)
\quad\Longleftrightarrow\quad
|P(y)|\le 2|y|.
\]

So the set of real \(y\) satisfying \(|P(y)|\le 2|y|\) is exactly the set of real solutions of \(y^2=P(0)\).

---

Claims:

- `s2c1` 标注判断=correct: Set \(x=0\).
- `s2c2` 标注判断=correct: If \(x=0\), then the left-hand inequality becomes 
\[
|y^2-P(0)|\le 0.
\]
- `s2c3` 标注判断=correct: \(|y^2-P(0)|\le 0\) is equivalent to 
\[
y^2=P(0).
\]
- `s2c4` 标注判断=correct: Hence, for every real \(y\), 
\[
y^2=P(0)
\quad\Longleftrightarrow\quad
|0-P(y)|\le 2|y|.
\]
- `s2c5` 标注判断=correct: \[y^2=P(0)
\quad\Longleftrightarrow\quad
|0-P(y)|\le 2|y|
\] is the same as 
\[
y^2=P(0)
\quad\Longleftrightarrow\quad
|P(y)|\le 2|y|.
\]
- `s2c6` 标注判断=correct: So the set of real \(y\) satisfying \(|P(y)|\le 2|y|\) is exactly the set of real solutions of \(y^2=P(0)\).

### s3

### Step 2: Interpret this according to the sign of \(P(0)\)

There are three cases.

#### Case 1: \(P(0)<0\)

Then the equation \(y^2=P(0)\) has no real solution. Therefore the condition from Step 1 says

\[
|P(y)|\le 2|y|
\]

has no real solution \(y\).

In particular, at \(y=0\) we must have

\[
|P(0)|\le 0
\]

false, which is indeed true since \(P(0)<0\). So negative values are at least not immediately excluded.

Claims:

- `s3c1` 标注判断=correct: Interpret this according to the sign of \(P(0)\).
- `s3c2` 标注判断=correct: There are three cases.
- `s3c3` 标注判断=correct: Case 1: \(P(0)<0\).
- `s3c4` 标注判断=correct: If \(P(0)<0\), then the equation \(y^2=P(0)\) has no real solution.
- `s3c5` 标注判断=correct: Therefore the condition from Step 1 says \(|P(y)|\le 2|y|\) has no real solution \(y\).
- `s3c6` 标注判断=correct: In particular, at \(y=0\) we must have 
\[
|P(0)|\le 0
\]
false.
- `s3c7` 标注判断=correct: The statement “\(|P(0)|\le 0\) is false” is indeed true since \(P(0)<0\).
- `s3c8` 标注判断=correct: So negative values are at least not immediately excluded.

### s4

#### Case 2: \(P(0)=0\)

Then \(y^2=P(0)\) has the unique solution \(y=0\). Thus Step 1 implies

\[
|P(y)|\le 2|y|
\quad\text{holds only at }y=0.
\]

But a polynomial with \(P(0)=0\) is divisible by \(y\): write

\[
P(y)=yQ(y)
\]

for some polynomial \(Q\). Then for \(y\) sufficiently close to \(0\),

\[
|P(y)|=|y||Q(y)|.
\]

Since \(Q\) is continuous, \(|Q(y)|\) is bounded near \(0\), so for all sufficiently small \(y\),

\[
|P(y)|\le 2|y|.
\]

Thus the inequality would hold for infinitely many \(y\) near \(0\), not only for \(y=0\). Contradiction.

Hence \(P(0)\ne 0\).

Claims:

- `s4c1` 标注判断=correct: Case 2: \(P(0)=0\).
- `s4c2` 标注判断=correct: If \(P(0)=0\), then \(y^2=P(0)\) has the unique solution \(y=0\).
- `s4c3` 标注判断=correct: Thus Step 1 implies \(|P(y)|\le 2|y|\) holds only at \(y=0\).
- `s4c4` 标注判断=correct: A polynomial with \(P(0)=0\) is divisible by \(y\).
- `s4c5` 标注判断=correct: Write \(P(y)=yQ(y)\) for some polynomial \(Q\).
- `s4c6` 标注判断=correct: For \(y\) sufficiently close to \(0\), \(|P(y)|=|y||Q(y)|\).
- `s4c7` 标注判断=correct: Since \(Q\) is continuous, \(|Q(y)|\) is bounded near \(0\).
- `s4c8` 标注判断=correct: So for all sufficiently small \(y\), \(|P(y)|\le 2|y|\).
- `s4c9` 标注判断=correct: Thus the inequality would hold for infinitely many \(y\) near \(0\), not only for \(y=0\).
- `s4c10` 标注判断=delete: Contradiction.
- `s4c11` 标注判断=correct: Hence \(P(0)\ne 0\).

### s5

#### Case 3: \(P(0)>0\)

Then \(y^2=P(0)\) has exactly two real solutions \(y=\pm \sqrt{P(0)}\). So Step 1 gives

\[
|P(y)|\le 2|y|
\quad\text{if and only if}\quad
y=\pm \sqrt{P(0)}.
\]

That means the polynomial

\[
P(y)^2-4y^2
\]

is \(\le 0\) exactly at those two points and \(>0\) elsewhere. Since this is a polynomial, each of those two points must be roots of even multiplicity unless the sign changes. But here it must be nonpositive only at isolated points, so both must actually be double roots and no others. Hence

\[
P(y)^2-4y^2 = c\,(y^2-P(0))^2
\]

for some constant \(c>0\).

Now compare degrees. The left-hand side is \(P(y)^2-4y^2\). For this to be a constant multiple of a quadratic squared, it must have degree \(4\), \(2\), or \(0\). This forces \(\deg P\le 2\). Let us instead use the values at the special points \(y=\pm \sqrt{P(0)}\): there,

\[
|P(y)|=2|y|=2\sqrt{P(0)}.
\]

So for \(a=\sqrt{P(0)}\),

\[
P(a)=\pm 2a,\qquad P(-a)=\pm 2a.
\]

Also, because \(P(y)^2-4y^2\) vanishes only at \(y=\pm a\), the factorization must be exact:

\[
P(y)^2-4y^2 = c(y^2-a^2)^2.
\]

Set \(y=0\). Then

\[
P(0)^2=c a^4.
\]

Since \(a^2=P(0)\), this gives \(P(0)^2=cP(0)^2\), so \(c=1\). Therefore

\[
P(y)^2-4y^2=(y^2-P(0))^2.
\]

Hence

\[
P(y)^2 = y^4 +(4-2P(0))y^2 + P(0)^2.
\]

Now set \(y=0\): this is consistent. To make the right side a perfect square polynomial, we must have

\[
y^4 +(4-2P(0))y^2 + P(0)^2 = (y^2+b)^2
\]

for some real \(b\), because there is no odd-power term. Expanding:

\[
(y^2+b)^2 = y^4+2by^2+b^2.
\]

Thus

\[
2b=4-2P(0),\qquad b^2=P(0)^2.
\]

So \(b=\pm P(0)\). If \(b=P(0)\), then

\[
2P(0)=4-2P(0)\quad\Rightarrow\quad 4P(0)=4\quad\Rightarrow\quad P(0)=1.
\]

If \(b=-P(0)\), then

\[
-2P(0)=4-2P(0)\quad\Rightarrow\quad 0=4,
\]

impossible. Therefore the only positive possibility is

\[
P(0)=1.
\]

---

Claims:

- `s5c1` 标注判断=correct: Case 3: \(P(0)>0\).
- `s5c2` 标注判断=correct: If \(P(0)>0\), then \(y^2=P(0)\) has exactly two real solutions \(y=\pm \sqrt{P(0)}\).
- `s5c3` 标注判断=correct: So Step 1 gives \(|P(y)|\le 2|y|\) if and only if \(y=\pm \sqrt{P(0)}\).
- `s5c4` 标注判断=correct: That means the polynomial \(P(y)^2-4y^2\) is \(\le 0\) exactly at those two points and \(>0\) elsewhere.
- `s5c5` 标注判断=correct: Since this is a polynomial, each of those two points must be roots of even multiplicity unless the sign changes.
- `s5c6` 标注判断=correct: Here it must be nonpositive only at isolated points, so both must actually be double roots and no others.
- `s5c7` 标注判断=correct: Hence \(P(y)^2-4y^2 = c\,(y^2-P(0))^2\) for some constant \(c>0\).
- `s5c8` 标注判断=delete: Now compare degrees.
- `s5c9` 标注判断=delete: The left-hand side is \(P(y)^2-4y^2\).
- `s5c10` 标注判断=delete: For this to be a constant multiple of a quadratic squared, it must have degree \(4\), \(2\), or \(0\).
- `s5c11` 标注判断=delete: This forces \(​\deg P\le 2\).
- `s5c12` 标注判断=correct: Let us instead use the values at the special points \(y=\pm \sqrt{P(0)}\).
- `s5c13` 标注判断=correct: At \(y=\pm \sqrt{P(0)}\), 
\[
|P(y)|=2|y|=2\sqrt{P(0)}.
\]
- `s5c14` 标注判断=correct: For \(a=\sqrt{P(0)}\), 
\[
P(a)=\pm 2a,\qquad P(-a)=\pm 2a.
\]
- `s5c15` 标注判断=correct: Because \(P(y)^2-4y^2\) vanishes only at \(y=\pm a\), the factorization must be exact: 
\[
P(y)^2-4y^2 = c(y^2-a^2)^2.
\]
- `s5c16` 标注判断=correct: Set \(y=0\).
- `s5c17` 标注判断=correct: Then \(P(0)^2=c a^4\).
- `s5c18` 标注判断=correct: Since \(a^2=P(0)\), this gives \(P(0)^2=cP(0)^2\).
- `s5c19` 标注判断=correct: So \(c=1\).
- `s5c20` 标注判断=correct: Therefore \(P(y)^2-4y^2=(y^2-P(0))^2\).
- `s5c21` 标注判断=correct: Hence 
\[
P(y)^2 = y^4 +(4-2P(0))y^2 + P(0)^2.
\]
- `s5c22` 标注判断=delete: Now set \(y=0\): this is consistent.
- `s5c23` 标注判断=correct: To make the right side a perfect square polynomial, we must have 
\[
y^4 +(4-2P(0))y^2 + P(0)^2 = (y^2+b)^2
\]
for some real \(b\), because there is no odd-power term.
- `s5c24` 标注判断=correct: Expanding: 
\[
(y^2+b)^2 = y^4+2by^2+b^2.
\]
- `s5c25` 标注判断=correct: Thus 
\[
2b=4-2P(0),\qquad b^2=P(0)^2.
\]
- `s5c26` 标注判断=correct: So \(b=\pm P(0)\).
- `s5c27` 标注判断=correct: If \(b=P(0)\), then \(2P(0)=4-2P(0)\).
- `s5c28` 标注判断=correct: From \(2P(0)=4-2P(0)\) it follows that \(4P(0)=4\).
- `s5c29` 标注判断=correct: From \(4P(0)=4\) it follows that \(P(0)=1\).
- `s5c30` 标注判断=correct: If \(b=-P(0)\), then \(-2P(0)=4-2P(0)\).
- `s5c31` 标注判断=correct: From \(-2P(0)=4-2P(0)\) it follows that \(0=4\).
- `s5c32` 标注判断=correct: The equation \(0=4\) is impossible.
- `s5c33` 标注判断=correct: Therefore the only positive possibility is \(P(0)=1\).

### s6

### Step 3: Collect all possibilities

From the three cases:

- \(P(0)<0\) is possible,
- \(P(0)=0\) is impossible,
- \(P(0)>0\) forces \(P(0)=1\).

So the only possible values are

\[
P(0)\in (-\infty,0)\cup\{1\}.
\]

Claims:

- `s6c1` 标注判断=correct: Collect all possibilities.
- `s6c2` 标注判断=correct: From the three cases, \(P(0)<0\) is possible.
- `s6c3` 标注判断=correct: From the three cases, \(P(0)=0\) is impossible.
- `s6c4` 标注判断=correct: From the three cases, \(P(0)>0\) forces \(P(0)=1\).
- `s6c5` 标注判断=correct: So the only possible values are 
\[
P(0)\in (-\infty,0)\cup\{1\}.
\]
- `s6c6` 标注判断=correct: Verification: If \(P(0)=0\), Step 2 showed a contradiction near \(0\).
- `s6c7` 标注判断=correct: Verification: If \(P(0)>0\), the equivalence at \(x=0\) forces exactly two solutions to \(|P(y)|\le 2|y|\), which leads uniquely to \(P(0)=1\).
- `s6c8` 标注判断=correct: Verification: Negative values are not excluded by the condition at \(x=0\), so they remain possible.

## Step Dependencies

```json
{
  "s2": [
    "s1c1"
  ],
  "s3": [
    "s2c6"
  ],
  "s4": [
    "s2c6"
  ],
  "s5": [
    "s2c6"
  ],
  "s6": [
    "s3c5",
    "s4c10",
    "s5c28"
  ]
}
```

## 模型生成输出 granite-4.1-8b

### gen_output 1

  

---

### 1.  Set \(x=0\).

Then the left inequality becomes  

\[
|y^2-P(0)|\le 0 \quad\Longrightarrow\quad y^2=P(0)\qquad (\forall y\in\mathbb R).
\]

The only way a real number equals \(y^2\) for **all** real \(y\) is for that number to be zero; otherwise the equality would fail for some \(y\). Hence  

\[
P(0)=0.
\]

---

### 2.  Set \(y=0\).

The right inequality becomes  

\[
|0^2-P(x)|\le 2|x|\quad\Longrightarrow\quad |P(x)|\le 2|x|\qquad (\forall x\in\mathbb R).
\]

Thus any polynomial satisfying the original property must obey \(|P(x)|\le 2|x|\) for all \(x\).  

---

### 3.  Verify consistency.

With \(P(0)=0\) the two inequalities are compatible: for \(x=0\) we have \(|y^2-P(0)|=|y^2|=y^2\le 0\) only when \(y=0\); this is exactly the condition required by the equivalence, because the right side \(|0^2-P(y)|=| -P(y) | = |P(y)|\) is also forced to be \(0\) for \(y=0\). No contradiction arises, and the condition \(|P(x)|\le 2|x|\) can indeed be satisfied by many polynomials (e.g., \(P(x)=0\) or \(P(x)=2x\)).

---

Since the only possible value for \(P(0)\) that makes the given equivalence hold for all real \(x,y\) is \(0\), we conclude

\[
\boxed{0}.
\]

### gen_output 2

 

### Step 2: Analyze the two possibilities for \(P(0)\)

#### (a) \(P(0)>0\)

Then \(y^2=P(0)\) has exactly two real solutions, \(y=\pm\sqrt{P(0)}\).  
Thus the inequality \(|P(y)|\le 2|y|\) must hold **only** for \(y=\pm\sqrt{P(0)}\).

But for any polynomial \(P\) with real coefficients, the inequality \(|P(y)|\le 2|y|\) is typically satisfied for infinitely many \(y\) near \(0\) (because \(P(0)=P(0)\) and \(P\) is continuous). In particular, if \(P(0)>0\), then \(|P(0)|>0\) and the inequality fails at \(y=0\) (since it would require \(0\le 0\) but \(|P(0)|>0\)). This contradiction shows that \(P(0)>0\) is impossible.

#### (b) \(P(0)<0\)

Then \(y^2=P(0)\) has **no** real solutions, so the left‑hand inequality \(|y^2-P(0)|\le 0\) is never satisfied. Consequently the equivalence would require that the right‑hand inequality \(|x^2-P(y)|\le 2|y|\) also be never satisfied for any \(x,y\). However, taking \(y=0\) gives \(|x^2-P(0)|\le 0\), i.e. \(x^2=P(0)\), which has no solution because \(P(0)<0\). Thus the right‑hand side is also never satisfied, and the equivalence holds vacuously for all \(x,y\). But we must also check consistency with the original condition for arbitrary \(x,y\); a deeper analysis (see Step 4) shows this leads to contradictions unless \(P(0)=0\).

Hence the only viable value among the possibilities is \(P(0)=0\).

### Step 3: Verify that \(P(0)=0\) indeed works

Assume \(P(0)=0\). The condition becomes

\[
|y^2-P(x)|\le 2|x|
\quad\Longleftrightarrow\quad
|x^2-P(y)|\le 2|y|.
\]

A natural candidate is a quadratic polynomial that aligns with the squares:

\[
P(t)=t^2.
\]

Then

\[
|y^2-P(x)| = |y^2-x^2| = |y-x|\,|y+x|,
\]

and similarly \(|x^2-P(y)| = |x-y|\,|x+y|\). Both expressions are symmetric in \(x\) and \(y\); therefore the inequalities are equivalent for all real \(x,y\). Moreover, \(P(0)=0^2=0\).

To see that no other polynomial can work, note that the condition forces the function \(f(t)=|t^2-P(t)|\) to satisfy \(f(x)\le 2|x|\) exactly when \(f(y)\le 2|y|\). This symmetry strongly restricts the form of \(P\); a detailed algebraic manipulation (omitted here for brevity) shows that any deviation from \(P(t)=t^2\) would break the equivalence for some pair \((x,y)\). Hence \(P(t)=t^2\) is the unique solution, and consequently \(P(0)=0\).

### Conclusion

The only possible value of \(P(0)\) consistent with the given property is  

\[
\boxed{0}.
\]

### gen_output 3

 However, we must check consistency with the original symmetric condition.

#### Case 2: \(P(0)=0\)

Now the equation \(y^2=P(0)\) gives \(y=0\) as the *only* solution. Thus the condition \(|P(y)|\le 2|y|\) must hold **only** at \(y=0\).

But at \(y=0\) the left side of the original equivalence is

\[
|0^2-P(x)| = |P(x)|
\le 2|x|
\]

which must be true for *all* real \(x\) (because the right side of the equivalence is \(|x^2-P(0)| = |x^2-P(x)|\)). Hence we obtain

\[
|P(x)|\le 2|x|\qquad(\forall x\in\mathbb R). \tag{1}
\]

Now substitute \(x=0\) in the original equivalence:

\[
|y^2-P(0)| = |y^2| = y^2 \le 2|y|
\quad\Longleftrightarrow\quad
|0^2-P(y)| = |P(y)|\le 2|y|.
\]

Condition (1) already guarantees the right‑hand side for all \(y\); the left‑hand side becomes \(y^2\le 2|y|\), which holds for all \(y\) because

\[
y^2\le 2|y| \iff |y|\le 2.
\]

Thus (1) forces \(|y|\le 2\) for every real \(y\), which is impossible. Therefore \(P(0)=0\) cannot satisfy the original condition.

#### Case 3: \(P(0)>0\)

Then \(y^2=P(0)\) has two real solutions \(y=\pm\sqrt{P(0)}\). Hence the condition \(|P(y)|\le 2|y|\) must hold **exactly** for those two values and for no others.

In particular, at \(y=0\) we must have the equivalence fail, i.e.

\[
|0^2-P(0)| = P(0) > 0 > 0,
\]
so the left side of the original equivalence is false, while the right side must also be false. This forces

\[
|0^2-P(0)| = P(0) > 2|0| = 0,
\]

which is automatically true. More importantly, consider \(x=\pm\sqrt{P(0)}\). For such an \(x\),

\[
|y^2-P(x)|\le 2|x|
\]

must hold for \(y=0\) if and only if \(|x^2-P(0)|\le 2|x|\). Since \(x^2=P(0)\), the right‑hand side becomes

\[
|P(0)-P(0)| = 0 \le 2|x|,
\]

which is true. Thus the left‑hand side must also be true for \(y=0\):

\[
|0-P(x)| = |P(x)|\le 2|x|.
\]

Hence for \(x=\pm\sqrt{P(0)}\) we have

\[
|P(\pm\sqrt{P(0)})|\le 2\sqrt{P(0)}. \tag{2}
\]

Now return to the original condition with arbitrary \(x,y\). Setting \(y=\sqrt{P(0)}\) (and using symmetry) yields

\[
|(\sqrt{P(0)})^2-P(x)| = |P(0)-P(x)|\le 2|x|
\quad\Longleftrightarrow\quad
|x^2-P(\sqrt{P(0)})|\le 2|x|.
\]

Because the inequality must hold for **all** \(x\), the only way to avoid contradictions is that the polynomial \(P\) itself satisfies

\[
P(x)=x^{2}+C
\]

for some constant \(C\). Substituting this form into the original inequality:

\[
|y^{2}-(x^{2}+C)| = |y^{2}-x^{2}-C| \le 2|x|
\quad\Longleftrightarrow\quad
|x^{2}-(y^{2}+C)| = |x^{2}-y^{2}-C| \le 2|y|.
\]

Both sides are identical after swapping \(x\) and \(y\), so the condition reduces to

\[
|y^{2}-x^{2}-C| \le 2|x| \quad\text{and}\quad |y^{2}-x^{2}-C| \le 2|y|
\]

for all real \(x,y\). This can hold for all \(x,y\) only when \(C=0\). Therefore

\[
P(x)=x^{2}
\]

is the unique polynomial satisfying the given property.

Finally, compute \(P(0)\):

\[
P(0)=0^{2}=0.
\]

But we already excluded \(P(0)=0\) in Case 2. Re‑examine the derivation: the earlier argument with \(x=0\) gave

\[
y^{2}=P(0)
\]

must hold **iff** \(|P(y)|\le 2|y|\). For \(P(x)=x^{2}\) we have \(P(0)=0\) and \(y^{2}=0\) only at \(y=0\). The inequality \(|P(y)|\le 2|y|\) for \(P(x)=x^{2}\) becomes \(|y^{2}|\le 2|y|\), i.e. \(y^{2}\le 2|y|\), which holds exactly for \(|y|\le 2\). Thus the equivalence fails for \(|y|>2\); however, the original condition must hold for **all** real \(x,y\). Checking \(x=3,\;y=3\) for \(P(x)=x^{2}\):

\[
|3^{2}-3^{2}|=0\le 6,\qquad |3^{2}-3^{2}|=0\le 6,
\]

so it works. But check \(x=3,\;y=3\) for \(P(x)=x^{2}+C\) with \(C\neq0\):

\[
|9-(9+C)| = | -C| = |C| \le 6,
\]
while
\[
|9-(9+C)| = | -C| = |C| \le 6,
\]
so any \(C\) with \(|C|\le 6\) would work for that pair, but not for all pairs. Therefore the only polynomial that works for **all** \(x,y\) is \(P(x)=x^{2}\).

Thus \(P(0)=0\).

Wait: earlier we derived \(P(x)=x^{2}+C\) and concluded \(C=0\). Let us re‑verify the symmetry directly:

Assume \(P\) satisfies the condition. Swap \(x\) and \(y\) in the inequality:

\[
|y^{2}-P(x)|\le 2|x|
\iff
|x^{2}-P(y)|\le 2|y|.
\]

Now replace \(x\) by \(-x\) and \(y\) by \(-y\). Because the right‑hand side involves \(|x|\) and \(|y|\), the inequality is even in each variable. Hence the function \(f(x)=P(x)-x^{2}\) must be odd: \(f(-x)=-f(x)\). Therefore \(P(x)=x^{2}+g(x)\) with \(g\) odd. Substitute into the inequality:

\[
|y^{2}-(x^{2}+g(x))| = |y^{2}-x^{2}-g(x)| \le 2|x|
\]
\[
\iff
|x^{2}-(y^{2}+g(y))| = |x^{2}-y^{2}-g(y)| \le 2|y|.
\]

Since the left side depends on \(x\) and the right side on \(y\) separately, the only way this holds for all \(x,y\) is that \(g\) is identically zero. Hence \(P(x)=x^{2}\).

Finally, compute \(P(0)=0^{2}=0\).

Thus the only possible value of \(P(0)\) is  

\[
\boxed{0}.
\]

### gen_output 4

  

#### Case 3: \(P(0)>0\)

Now \(y^2=P(0)\) gives two real solutions \(y=\pm\sqrt{P(0)}\). Consequently, Step 1 tells us that

\[
|P(y)|\le 2|y|
\]

holds **exactly** for \(y=\pm\sqrt{P(0)}\) and fails for all other \(y\).

Assume \(P(0)>0\) and write \(P(y)=a_0+yR(y)\) with \(a_0=P(0)>0\).  
For \(y=\pm\sqrt{P(0)}\) we must have \(|P(y)|\le 2|y|\). Evaluating at \(y=\sqrt{P(0)}\) gives

\[
\bigl|\,a_0+\sqrt{P(0)}\,R(\sqrt{P(0)})\,\bigr|\le 2\sqrt{P(0)}.
\]

Because \(a_0>0\), the left‑hand side is at least \(a_0\); hence

\[
a_0\le 2\sqrt{P(0)}.
\tag{1}
\]

A similar bound follows from the other root.  

Now use the original equivalence for arbitrary \(x\) and \(y\). Substituting \(y=\sqrt{P(0)}\) into the condition yields

\[
\bigl|\,P(x)\bigr|\le 2|x|
\quad\Longleftrightarrow\quad
\bigl|\,x^{2}-P(\sqrt{P(0)})\bigr|\le 2\sqrt{P(0)}.
\]

The right‑hand inequality is independent of \(x\); for large \(|x|\) the left‑hand side \(|x^{2}-P(\sqrt{P(0)})|\) grows like \(|x|^{2}\), which eventually exceeds \(2\sqrt{P(0)}\). Thus the equivalence cannot hold for all real \(x\) unless the polynomial is of very low degree.  

A direct check shows that the only polynomial satisfying the required identity for all \(x,y\) is

\[
P(x)=x^{2}.
\]

Indeed, for \(P(x)=x^{2}\),

\[
|y^{2}-x^{2}|=|(y^{2}-x^{2})|\le 2|x|
\quad\Longleftrightarrow\quad
|x^{2}-y^{2}|\le 2|y|,
\]

which is symmetric in \(x\) and \(y\); hence the equivalence holds.  
For this polynomial \(P(0)=0\), contradicting our earlier exclusion of \(P(0)=0\).  

Thus the only remaining possibility is that our earlier deduction from Step 1 was incomplete. Revisiting Step 1 with \(P(0)>0\) forces the inequality \(|P(y)|\le 2|y|\) to hold for **all** real \(y\), because otherwise the equivalence would fail for some \(x\neq0\).  

If \(|P(y)|\le 2|y|\) for every \(y\), then \(P(y)=yQ(y)\) with \(|Q(y)|\le 2\) for all \(y\). The only polynomials bounded by a constant on \(\mathbb R\) are constant polynomials. Hence \(Q\) is constant, say \(Q\equiv c\). Therefore

\[
P(y)=c\,y.
\]

Plugging this into the original condition:

\[
|y^{2}-c x|\le 2|x|
\quad\Longleftrightarrow\quad
|x^{2}-c y|\le 2|y|
\qquad (\forall x,y).
\]

Letting \(x\to0\) gives \(|y^{2}|\le0\), which forces \(y=0\). Hence the only possible constant is \(c=0\), yielding \(P(x)=0\).

Finally, \(P(x)=0\) indeed satisfies the given equivalence, and

\[
P(0)=0.
\]

Thus the only admissible value of \(P(0)\) is \(0\).

\[
\boxed{0}
\]

### gen_output 5

 
### Step 3: Verify the polynomial for \(P(0)=1\)

From \(P(y)^2=(y^2-1)^2\) and the sign analysis above, we take

\[
P(y)=y^2-1.
\]

Check the original condition:

\[
|y^2-P(x)| = |y^2-(x^2-1)| = |(y^2- x^2)+1|.
\]

For any \(x,y\),

\[
|x^2-P(y)| = |x^2-(y^2-1)| = |(x^2- y^2)+1|,
\]

which is exactly the same expression. Hence

\[
|y^2-P(x)|\le 2|x|
\quad\Longleftrightarrow\quad
|x^2-P(y)|\le 2|y|
\]

holds for all real \(x,y\). Thus \(P(0)=1\) is admissible.

### Step 4: No other values

We have shown:
- \(P(0)<0\) leads to a contradiction with the requirement that the inequality be satisfied at \(x=0\) (it would require \(y^2=P(0)\) for some \(y\), which is impossible).
- \(P(0)=0\) forces the polynomial to be divisible by \(y\), which makes the inequality hold near \(y=0\) contrary to the “if and only if” condition.
- \(P(0)>0\) forces \(P(0)=1\) as derived, and the polynomial \(P(y)=y^2-1\) satisfies the condition.

Hence the only possible value of \(P(0)\) is

\[
\boxed{1}.
\]

### gen_output 6

 

However, we must verify whether any polynomial with \(P(0)<0\) indeed satisfies the original condition. Suppose \(P(0)<0\). Then the inequality \(|y^2-P(0)|\le 2|x|\) with \(x=0\) would require \(|y^2+|P(0)||\le 0\), impossible for any \(y\neq 0\). Thus the “if and only if” condition would fail for \(y\neq 0\). Therefore the only admissible values are those that work for all \(x,y\).

Checking the candidate \(P(0)=1\):
Take \(P(x)=x^2+1\) (which indeed has \(P(0)=1\)). Then

\[
|y^2-P(x)| = |y^2-(x^2+1)| = |y^2-x^2-1|,
\]
\[
|x^2-P(y)| = |x^2-(y^2+1)| = |x^2-y^2-1| = |y^2-x^2-1|.
\]

Thus \(|y^2-P(x)|\le 2|x|\) is equivalent to \(|x^2-P(y)|\le 2|y|\) for all real \(x,y\). Hence \(P(x)=x^2+1\) satisfies the condition, confirming that \(P(0)=1\) is attainable.

Consequently, the only possible value of \(P(0)\) is

\[
\boxed{1}.
\]

## Judge Prompt 路由摘要

|route|rows|
|---|---:|
|holistic|4|
|pairwise|3|
|selfjudge_with_reference|27|
|selfjudge_without_reference|4|

完整 judge prompt rows 请看对应 `per_sample_json/*.json` 的 `stage_04_remote_actual_judge_cache_rows`，或总表 `stage_04_judge_cache_granite4_1_8b_slice.jsonl`。
