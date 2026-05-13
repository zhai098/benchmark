# 样本 1: q-1 / WangZX / sample 2

## 定位信息

- annotation_uid: `q-1__WangZX__dev-1776827054895-o7cnww__sample_2`
- 原始标注 detail_path: `annotation_app/data/annotations/WangZX/dev-1776827054895-o7cnww/q-1.detail.json`
- official_path: `official_annotation_data_p2/official_annotation_data_p2_part_01.jsonl`
- clean input slice line: 1
- remote gen_only source line: 1

## 全链路计数

- reference steps: 5
- reference claims: 21
- generate.py actual prompts: 5
- Kimi/Moonshot prompt-pack prompts: 4
- model gen_output entries: 5
- judge cache rows: 23
- judge route counts: `{'pairwise': 1, 'holistic': 3, 'selfjudge_without_reference': 3, 'selfjudge_with_reference': 16}`
- local repack row-count match: `True`

## 题目

A tournament is a directed graph for which every (unordered) pair of vertices has a single directed edge from one vertex to the other.  Let us define a proper directed-edge-coloring to be an assignment of a color to every (directed) edge, so that for every pair of directed edges $\overrightarrow{uv}$ and $\overrightarrow{vw}$, those two edges are in different colors.  Note that it is permissible for $\overrightarrow{uv}$ and $\overrightarrow{uw}$ to be the same color.  The directed-edge-chromatic-number of a tournament is defined to be the minimum total number of colors that can be used in order to create a proper directed-edge-coloring.  For each $n$, determine the minimum directed-edge-chromatic-number over all tournaments on $n$ vertices.

## 参考答案 / 标准解

\lceil \log_2 n \rceil

## 被保留的正确 sample 解答

Fix a tournament \(T\) whose directed edges are properly colored with \(k\) colors.

For each vertex \(v\), let
\[
S(v)=\{\text{colors appearing on edges directed out of }v\}.
\]
Because the coloring is proper, no color can appear on both an incoming and an outgoing edge at the same vertex: if \(u\to v\) and \(v\to w\) had the same color, this would violate the rule.

Now take distinct vertices \(x\neq y\). Exactly one of \(x\to y\) or \(y\to x\) holds; assume \(x\to y\) has color \(c\). Then \(c\in S(x)\). Also \(c\notin S(y)\), because if \(c\in S(y)\) then there exists some edge \(y\to z\) of color \(c\), and the pair \(x\to y\) and \(y\to z\) would be a forbidden same-colored consecutive pair. Hence \(S(x)\neq S(y)\).

Thus the sets \(S(v)\) are pairwise distinct subsets of a \(k\)-element color set, so
\[
n \le 2^k \quad\Rightarrow\quad k\ge \lceil \log_2 n\rceil.
\]
This lower bound holds for every tournament on \(n\) vertices, so the minimum possible directed-edge-chromatic-number is at least \(\lceil \log_2 n\rceil\).

For the matching upper bound, let \(k=\lceil \log_2 n\rceil\). Consider \(2^k\) vertices labeled by all binary strings \(x=(x_1,\dots,x_k)\in\{0,1\}^k\). For two distinct strings \(x,y\), let \(i\) be the smallest index where they differ. Orient the edge from the vertex with \(1\) in coordinate \(i\) to the vertex with \(0\) in coordinate \(i\), and color this directed edge with color \(i\). This produces a tournament.

To check properness, suppose \(x\to y\) and \(y\to z\) both have color \(i\). Then by definition of the coloring we must have \(x_i=1,y_i=0\) from \(x\to y\), and also \(y_i=1,z_i=0\) from \(y\to z\), a contradiction. Hence no such consecutive pair shares a color, so the coloring is proper and uses at most \(k\) colors. Taking any \(n\) of these \(2^k\) vertices and restricting to the induced subtournament preserves properness and uses at most \(k\) colors.

Therefore the minimum directed-edge-chromatic-number over all tournaments on \(n\) vertices equals \(\lceil \log_2 n\rceil\).

## 标注者决策摘要

_无_

## 标注 steps 与 claims

### s1

Fix a tournament \(T\) whose directed edges are properly colored with \(k\) colors.

For each vertex \(v\), let
\[
S(v)=\{\text{colors appearing on edges directed out of }v\}.
\]
Because the coloring is proper, no color can appear on both an incoming and an outgoing edge at the same vertex: if \(u\to v\) and \(v\to w\) had the same color, this would violate the rule.

Claims:

- `s1c1` 标注判断=correct: Fix a tournament \(T\) whose directed edges are properly colored with \(k\) colors.
- `s1c2` 标注判断=correct: For each vertex \(v\), let 
\[
S(v)=\{\text{colors appearing on edges directed out of }v\}.
\]
- `s1c3` 标注判断=correct: Because the coloring is proper, no color can appear on both an incoming and an outgoing edge at the same vertex: if \(u\to v\) and \(v\to w\) had the same color, this would violate the rule.

### s2

Now take distinct vertices \(x\neq y\). Exactly one of \(x\to y\) or \(y\to x\) holds; assume \(x\to y\) has color \(c\). Then \(c\in S(x)\). Also \(c\notin S(y)\), because if \(c\in S(y)\) then there exists some edge \(y\to z\) of color \(c\), and the pair \(x\to y\) and \(y\to z\) would be a forbidden same-colored consecutive pair. Hence \(S(x)\neq S(y)\).

Thus the sets \(S(v)\) are pairwise distinct subsets of a \(k\)-element color set, so
\[
n \le 2^k \quad\Rightarrow\quad k\ge \lceil \log_2 n\rceil.
\]
This lower bound holds for every tournament on \(n\) vertices, so the minimum possible directed-edge-chromatic-number is at least \(\lceil \log_2 n\rceil\).

Claims:

- `s2c1` 标注判断=correct: Now take distinct vertices \(x\neq y\). Exactly one of \(x\to y\) or \(y\to x\) holds; assume \(x\to y\) has color \(c\).
- `s2c2` 标注判断=correct: Then \(c\in S(x)\).
- `s2c3` 标注判断=correct: Also \(c\notin S(y)\), because if \(c\in S(y)\) then there exists some edge \(y\to z\) of color \(c\), and the pair \(x\to y\) and \(y\to z\) would be a forbidden same-colored consecutive pair.
- `s2c4` 标注判断=correct: Hence \(S(x)\neq S(y)\).
- `s2c5` 标注判断=correct: Thus the sets \(S(v)\) are pairwise distinct subsets of a \(k\)-element color set.
- `s2c6` 标注判断=correct: \[
n \le 2^k.
\]
- `s2c7` 标注判断=correct: \[
k\ge \lceil \log_2 n\rceil.
\]
- `s2c8` 标注判断=correct: This lower bound holds for every tournament on \(n\) vertices, so the minimum possible directed-edge-chromatic-number is at least \(\lceil \log_2 n\rceil\).

### s3

For the matching upper bound, let \(k=\lceil \log_2 n\rceil\). Consider \(2^k\) vertices labeled by all binary strings \(x=(x_1,\dots,x_k)\in\{0,1\}^k\). For two distinct strings \(x,y\), let \(i\) be the smallest index where they differ. Orient the edge from the vertex with \(1\) in coordinate \(i\) to the vertex with \(0\) in coordinate \(i\), and color this directed edge with color \(i\). This produces a tournament.

Claims:

- `s3c1` 标注判断=correct: For the matching upper bound, let \(k=\lceil \log_2 n\rceil\).
- `s3c2` 标注判断=correct: Consider \(2^k\) vertices labeled by all binary strings \(x=(x_1,\dots,x_k)\in\{0,1\}^k\).
- `s3c3` 标注判断=correct: For two distinct strings \(x,y\), let \(i\) be the smallest index where they differ.
- `s3c4` 标注判断=correct: Orient the edge from the vertex with \(1\) in coordinate \(i\) to the vertex with \(0\) in coordinate \(i\), and color this directed edge with color \(i\).
- `s3c5` 标注判断=correct: This produces a tournament.

### s4

To check properness, suppose \(x\to y\) and \(y\to z\) both have color \(i\). Then by definition of the coloring we must have \(x_i=1,y_i=0\) from \(x\to y\), and also \(y_i=1,z_i=0\) from \(y\to z\), a contradiction. Hence no such consecutive pair shares a color, so the coloring is proper and uses at most \(k\) colors. Taking any \(n\) of these \(2^k\) vertices and restricting to the induced subtournament preserves properness and uses at most \(k\) colors.

Claims:

- `s4c1` 标注判断=correct: To check properness, suppose \(x\to y\) and \(y\to z\) both have color \(i\).
- `s4c2` 标注判断=correct: Then by definition of the coloring we must have \(x_i=1,y_i=0\) from \(x\to y\), and also \(y_i=1,z_i=0\) from \(y\to z\), a contradiction.
- `s4c3` 标注判断=correct: Hence no such consecutive pair shares a color, so the coloring is proper and uses at most \(k\) colors.
- `s4c4` 标注判断=correct: Taking any \(n\) of these \(2^k\) vertices and restricting to the induced subtournament preserves properness and uses at most \(k\) colors.

### s5

Therefore the minimum directed-edge-chromatic-number over all tournaments on \(n\) vertices equals \(\lceil \log_2 n\rceil\).

Claims:

- `s5c1` 标注判断=correct: Therefore the minimum directed-edge-chromatic-number over all tournaments on \(n\) vertices equals \(\lceil \log_2 n\rceil\).

## Step Dependencies

```json
{
  "s2": [
    "s1c3"
  ],
  "s4": [
    "s3c4",
    "s3c3"
  ],
  "s5": [
    "s2c6",
    "s4c4"
  ]
}
```

## 模型生成输出 granite-4.1-8b

### gen_output 1

 Hence each color in \(S(v)\) appears on **exactly one** outgoing edge from \(v\).

Thus the number of colors used in \(T\) equals the maximum, over all vertices \(v\), of \(|S(v)|\). Consequently,
\[
\chi^{*}(T)=\max_{v\in V(T)}|S(v)|.
\]

Now consider any tournament on \(n\) vertices. Pick any vertex \(v\). It has out‑degree
\[
d^{+}(v)\in\{0,1,\dots ,n-1\}.
\]
All \(d^{+}(v)\) outgoing edges from \(v\) must receive distinct colors, because two edges incident to the same vertex cannot share a color (otherwise a conflict would arise with the intermediate vertex). Therefore,
\[
|S(v)|\ge d^{+}(v).
\]
Summing over all vertices,
\[
\max_{v}|S(v)|\ge \max_{v} d^{+}(v).
\]

In any tournament there exists a vertex with out‑degree at least \(\frac{n-1}{2}\); however, the bound we need is the **minimum** possible value of \(\max_{v}|S(v)|\) over all tournaments. To minimize this maximum, we should make the out‑degrees as balanced as possible. The most balanced tournament is the **transitive tournament**, where the vertices can be ordered \(v_{1},v_{2},\dots ,v_{n}\) such that every edge is oriented from a lower‑indexed vertex to a higher‑indexed vertex.

In the transitive tournament:
- Vertex \(v_i\) has out‑degree \(d^{+}(v_i)=n-i\).
- The largest out‑degree occurs at \(v_1\) and equals \(n-1\).

Since each outgoing edge at \(v_1\) must receive a distinct color, any proper directed‑edge‑coloring of this tournament requires at least \(n-1\) colors. The natural coloring—assign a different color to each edge incident to \(v_1\)—achieves exactly \(n-1\) colors, so the transitive tournament attains this lower bound.

Therefore, for every tournament on \(n\) vertices,
\[
\chi^{*}(T)\ge n-1,
\]
and there exists a tournament (the transitive one) with \(\chi^{*}=n-1\). Hence the minimum possible directed‑edge‑chromatic number over all tournaments on \(n\) vertices is

\[
\boxed{n-1}.
\]

### gen_output 2

  
To show that this bound is attainable, construct a tournament by orienting the edges according to the binary representation of the vertices \(0,1,\dots ,n-1\). Label each vertex by its binary string of length \(\lceil \log_2 n\rceil\). For vertices \(i\) and \(j\), compare their binary strings lexicographically; the edge goes from the string that is smaller to the larger one. Color the edge \(i\to j\) with the color corresponding to the first binary position where the strings differ. This uses exactly \(\lceil \log_2 n\rceil\) colors, and any pair of consecutive edges \(i\to j\) and \(j\to k\) differ in at least one of those positions, guaranteeing different colors. Hence the bound is tight.

Consequently, the minimum directed‑edge‑chromatic‑number over all tournaments on \(n\) vertices is  

\[
\boxed{\lceil \log_2 n \rceil}.
\]

### gen_output 3

 Any two consecutive edges \(\overrightarrow{uv}\) and \(\overrightarrow{vw}\) use colors determined by the differing bits of \(u\) and \(v\), and of \(v\) and \(w\); because \(v\) shares the prefix up to the first differing bit with exactly one of \(u,w\), the two colors are distinct. Hence this coloring uses exactly \(k\) colors, achieving the lower bound.

Therefore, for every \(n\ge 1\), the minimum directed-edge-chromatic-number over all tournaments on \(n\) vertices equals \(\lceil \log_2 n\rceil\).

\[
\boxed{\lceil \log_2 n\rceil}
\]

### gen_output 4

 Therefore a tournament on \(n\) vertices can be properly edge‑colored with \(\lceil \log_2 n\rceil\) colors.

Combining the lower and upper bounds, the minimum directed‑edge‑chromatic‑number over all tournaments on \(n\) vertices is exactly \(\boxed{\lceil \log_2 n\rceil}\).

### gen_output 5

 

\[
\boxed{\lceil \log_2 n\rceil}
\]

## Judge Prompt 路由摘要

|route|rows|
|---|---:|
|holistic|3|
|pairwise|1|
|selfjudge_with_reference|16|
|selfjudge_without_reference|3|

完整 judge prompt rows 请看对应 `per_sample_json/*.json` 的 `stage_04_remote_actual_judge_cache_rows`，或总表 `stage_04_judge_cache_granite4_1_8b_slice.jsonl`。
