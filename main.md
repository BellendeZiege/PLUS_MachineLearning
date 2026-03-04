# Statistical Learning Theory

**Univ.-Prof. Dr. Roland Kwitt** — Summer Semester 2025

Book: *Understanding Machine Learning* — Shai Shalev-Shwartz and Shai Ben-David

Online Content: http://rkwitt.org (see Teaching) and https://github.com/rkwitt/teaching

---

## 1. Introduction

### 1.1 Practice Problem

Suppose our goal is to classify a dataset of papayas. We want to find out which papaya is tasty and which is not. Therefore, we need a function that can assert such property given a number of input features (e.g. weight, color). This function, further called *hypothesis*, will take a two-dimensional vector and supply us with an output indicating whether a papaya is tasty or not by the labels 0 (not tasty) and 1 (tasty).

**Table 1: Papaya Classification — Training Dataset**

| Observation | Weight [g] | Color [0,1] | Tasty ∈ {0,1} |
|-------------|-----------|-------------|----------------|
| Papaya 1    | 100       | 0.1         | yes (1)        |
| ...         | ...       | ...         | ...            |
| Papaya N    | 500       | 0.8         | no (0)         |

The information in the table is called the *training data* $S=\{(x_1, y_1),\, \ldots \, , (x_n, y_n)\}$. In our example $x_i \in \mathbb{R}^2$ and $y_i \in \{0,1\}$. Based on that, we want to find:

$$h: \mathbb{R} \times \mathbb{R} \to \{0,1\}$$

### 1.2 Learning

A *learner* receives $S$ and outputs a hypothesis $h$ (some algorithm).

We will make two assumptions initially:

- All the $x_i$'s are drawn *independently and identically distributed* (i.i.d.) from some (unknown) distribution $\mathcal{D}$ over our domain set $X$.
- The $x_i$'s are labeled by some (unknown) function $f: X \to Y$ (labeling function). This means $S=\{(x_1, f(x_1)),\, \ldots \, , (x_n, f(x_n))\}$.

We care about whether the hypothesis makes an incorrect prediction (with respect to $f$). These are all the points $x$ in our domain $X$, where the hypothesis $h$ differs from the labeling function $f$. In our example $X=\mathbb{R}^2$, so let $A \in \mathcal{P}(\mathbb{R}^2)$:

$$A=\{x \in X \mid h(x) \neq f(x)\} \quad \text{(misclassifications)}$$

1. **Domain set** $X$: we call $x \in X$ an instance.
2. **Label set** $Y$, e.g. $Y=\{0,1\}$.
3. **Training set** $S=\{(x_1,y_1),\, \ldots \, , (x_m, y_m)\}$ with $(x_i, y_i) \in X \times Y := Z$.
4. A learner receives $S$ and outputs $h: X \to Y$ (e.g. a hypothesis).

### 1.3 Measure Problem

Find $\mu: \mathcal{P}(\mathbb{R}^n) \to [0,\infty]$ with the following properties:

I. $A_i \in \mathcal{P}(\mathbb{R}^2),\, i \in \mathbb{N}$ pairwise disjoint $\Rightarrow \mu\!\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty \mu(A_i)$

II. If $A,B \in \mathcal{P}(\mathbb{R}^2)$ are congruent: $\mu(A) = \mu(B)$

III. $\mu([0,1]^n)=1$

This problem is not solvable for all $n \in \mathbb{N}$ (unless we'd accept $\mu \equiv 0$). We will constrain $\mathcal{A}$ to be an element of a $\sigma$-Algebra over $X$. Later, we will talk about quantities of the form:

$$\underset{x \sim \mathcal{D}}{\mathbb{P}}\big[h(x) \neq f(x)\big] = \mathcal{D}\big(\{x \in X: h(x) \neq f(x)\}\big) \qquad \text{(Generalization error)}$$

---

## 2. Statistical Learning Theory

More formally, the data available to us comes in the form:

$$S=\big((x_1, y_1),\, \ldots \, , (x_n, y_n)\big)$$

The *training set* is based on the *domain set* $X$, where we call $x \in X$ an instance, and $Y$ the label set (e.g. $Y=\{0,1\}$). Thus, a learner receives $S$ and outputs a *hypothesis* $h: X \to Y$. In the papaya example, the domain is $x_i \in \mathbb{R}^2$ and the label set $y_i \in \{0,1\}$.

We make two assumptions initially:

1. All the $x_i$'s are drawn independently and identically distributed (i.i.d.) from some (unknown) distribution $\mathcal{D}$ over the domain $X$.
2. The $x_i$'s are labeled by some (unknown) function $f: X \to Y$, called the labeling function. This means $S=\big((x_1, y_1=f(x_1)),\, \ldots \, , (x_n, y_n = f(x_n))\big)$.

We care about all the points $x$ in our domain $X$, where the hypothesis $h(x)$ differs from the true labeling function $f(x)$:

$$A = \{x \in X: h(x) \neq f(x)\}$$

---

## 3. No Free Lunch

There is no generally perfect method. What we really do, when restricting the search space (e.g. for ERM) to a class $\mathcal{H}$, is to hope that one $h \in \mathcal{H}$ has *low error*. This is a form of *prior knowledge*. The question is: is this really necessary?

What if there is some form of *universal learner*? That is, a learner that succeeds on any given task (defined by distribution $\mathcal{D}$ over $X \times Y$). By *succeed* we mean finding a $h: X \to Y$ with low risk $L_D(h)$.

We will show, for the task of binary classification, that no such learner exists. In particular, we show that there exists a distribution such that upon receiving $m$ samples from that distribution, the learner outputs a hypothesis with high risk (generalization error), while another learner exists with small risk. So every learner has tasks on which it fails, while others succeed.

> **Theorem (No Free Lunch).** Let $A$ be any learning algorithm for binary classification with respect to the 0-1 loss over a domain $X$. Further, let $m < \frac{|X|}{2}$. Then, there exists a distribution $\mathcal{D}$ over $X \times \{0,1\}$ such that:
> 1. $\exists\, f: X \to \{0,1\}$ with $L_D(f) = 0$
> 2. $\mathcal{D}^m\!\left(\{S: L_D(A(S)) \geq \tfrac{1}{8}\}\right) \geq \tfrac{1}{7}$ (with $|S|=m$)

**Proof.** Let $C \subset X$ of size $2m$. The number of possible labellings of $C$ is $T=2^{2m}$. We denote the functions that realize all these labellings by $f_1, \ldots, f_T$. For each $f_i$, we define:

$$\mathcal{D}_i(\{(x,y)\}) = \begin{cases} \frac{1}{|C|} & \text{if } y=f_i(x) \\ 0 & \text{else}\end{cases}$$

We see that $L_D(f_i) = 0$ (by construction).

We aim to show that $\exists\, \mathcal{D}$ over $X \times \{0,1\}$ and a function $f: X \to \{0,1\}$ such that $L_D(f)=0$, but:

$$\underset{S \sim D}{\mathbb{E}}\Big[L_D(A(S))\Big] \geq \text{const.}$$

To that end, we aim to prove:

$$\max_{i \in \{1,2,3,\ldots\}}\left(\underset{S \sim D}{\mathbb{E}}\Big[L_D(A(S))\Big] \geq \text{const.}\right)$$

**Step 1.** We know that there are $(2m)^m$ possible training sets of size $m$.

*Example:* $C=\{c_1, c_2\}$, $m=1$: We have $\{c_1\}$ or $\{c_2\}$. $\square$

---

## 4. Lecture 20.05.25

**Question.** Is there a universal learner? By a universal learner we mean a learner without any prior knowledge of the learning task (given by $\mathcal{D}$), but that can be challenged by any task and still returns $A(S)$ with low generalization error $L_D(A(S))$.

This question is answered by the following theorem:

> **Theorem (No Free Lunch).** Let $A$ be a learning algorithm for the task of binary classification with respect to the 0-1 loss, over domain $X$. Also, let $m \in \mathbb{N}$ be any number smaller than $\tfrac{|X|}{2}$ (representing the size of $S$). Then, there exists a distribution $\mathcal{D}$ over $X \times \{0,1\}$, such that: *(see enumeration in §3).*
>
> This means that whatever you see on the training set, you can see anything on the test set. Without any assumption about the learning task, there is no learning.
>
> **Interpretation.** Condition 1 means that the task can be learned successfully (perfectly) by another learner (e.g. $\mathcal{H}=\{f\}$), and condition 2 means that $A$ fails on that task.

> **Corollary.** Let $X$ be infinitely large and $\mathcal{H}$ be the class of all functions from $X \to \{0,1\}$. Then $\mathcal{H}$ is not PAC learnable.

### 4.1 Vapnik–Chervonenkis Dimension (VC-Dimension)

**Definition.** Let $\mathcal{H}$ be a class of functions from $X \to \{0,1\}$ and let $C \subset X$, $C=\{c_1, \ldots, c_n\}$. We define the *restriction of $H$ to the set $C$* as:

$$H_C = \Big\{(h(c_1), h(c_2), \ldots , h(c_n)): h \in \mathcal{H}\Big\}$$

**Example.** $C=\{c_1\}$, $c_1 \in \mathbb{R}$ and let $H^{\text{threshold}}$ be the class of thresholds on the real line.

For $C=\{c_1\}$: $H_C^{\text{thr}} = \{(1), (0)\} \Rightarrow |H_C^{\text{thr}}| = 2$

For $C=\{c_1, c_2\}$, $c_1 \leq c_2$:

$$H_C^{\text{thr}} = \{(1,1), (0,0), (1,0)\} \Rightarrow |H_C| = 3$$

*(Note: something is missing here — see original notes.)*

But we would have $2^2 = 4$ possible labellings of $C = \{c_1, c_2\}$.

**Definition (Shattering).** $H$ *shatters* a finite set $C$ of size $m$ if $|H_C|=2^m$.

**Claim.** The class $H^{\text{thr}}$ on $X=\mathbb{R}$ is *PAC learnable* with:

$$m_{H^{\text{thr}}}(\varepsilon, \delta) \leq \left\lceil \log\!\left(\frac{2}{\delta}\right) \cdot \frac{1}{\varepsilon} \right\rceil$$

We assume realizability, e.g. $\exists\, h^* \in H^{\text{thr}}$ such that $L_{D,f}(h^*)=0$.

We let $a_0 \in \mathbb{R}$ be such that:

$$\mathbb{D}\Big(\{x \in \mathbb{R}: x \in (a_0, a^*)\}\Big) = \varepsilon$$

and let $a_1 \in \mathbb{R}$ be such that:

$$\mathbb{D}\Big(\{x \in \mathbb{R}: x \in (a^*, a_1)\}\Big) = \varepsilon$$

**Special cases:**
$$\mathbb{D}\Big(\{x \in \mathbb{R}: x \in (a_0, a^*)\}\Big) < \varepsilon \Rightarrow \text{set } a_0 = -\infty$$
$$\mathbb{D}\Big(\{x \in \mathbb{R}: x \in (a^*, a_1)\}\Big) < \varepsilon \Rightarrow \text{set } a_1 = +\infty$$

Next, we define an ERM algorithm (needed for PAC). We pick $b_0 \in \mathbb{R}$ and $b_1 \in \mathbb{R}$ from $S=((x_1, y_1), \ldots , (x_n, y_n))$ as:

$$b_0 = \max\{x: (x,1) \in S\}$$
$$b_1 = \min\{x: (x,0) \in S\}$$

Then pick any threshold within $(b_0, b_1)$ and call the corresponding hypothesis $h_S$. We see that $h_S$ always has 0 empirical error on $S$.

By our construction of $a_0$ and $a_1$, for $h_S$ (ERM hypothesis, threshold $a_s$) to have generalization error $L_{D,f}(h_S) \leq \varepsilon$, it suffices that:

(a) $b_1 \leq a_1$, AND  
(b) $b_0 \geq a_0$

---

## 5. Lecture 27.05.2025

Writing this down formally:

$$\mathbb{P}\Big[L_{D,S}(h) > \varepsilon\Big] \leq \mathbb{P}\Big[(b_0 < a_0) \wedge (b_1 > a_1)\Big] \quad \text{(Union Bound)}$$
$$\leq \mathbb{P}\Big[b_0 < a_0\Big] + \mathbb{P}\Big[b_1 > a_1\Big]$$

The upper bound $\mathbb{P}[L_{D,f}(h) > \varepsilon]$ — we have to check when $b_0 < a_0$ (or $b_1 > a_1$). Answer: when there is no data-point in $S$ that is labeled 1, such that $x \in (a_0, a^*)$.

We know that $D(\{x \in \mathbb{R}: x \in (a_0, a^*)\}) = \varepsilon$ (by construction of $a_0$), hence *not seeing* a data-point in $(a_0, a^*)$ has probability of $1-\varepsilon$. Consequently, *not seeing* a data-point in $m$ i.i.d. samples inside $(a_0, a^*)$ has probability $(1-\varepsilon)^m$.

Overall:

$$\mathbb{P}[L_{D,S}(h) > \varepsilon] \leq 2 \cdot (1-\varepsilon)^m \leq 2\exp(-\varepsilon m)$$

> **Sample complexity:**
> $$m > \frac{1}{\varepsilon} \log(2 / \delta)$$

Thresholds are PAC-learnable.

We have seen that *finiteness* of $H$ is **sufficient**, but not **necessary** for PAC learnability.

> **Definition (VC-Dimension).** The VC-Dimension of $H$ (a class of functions $X \to \{0,1\}$), written as $\text{VC}(H)$, is the maximal size of a set $C \subset X$ that is shattered by $H$.

> **Theorem.** Let $H$ be a class of functions from $X \to \{0,1\}$. If $H$ has infinite VC-Dimension, then $H$ is not PAC-learnable. This follows immediately from the No Free Lunch theorem.

> **Definition (Growth function).** Let $H$ be a class of functions from $X \to \{0,1\}$. The *growth function* $\tau_H: \mathbb{N}_0 \to \mathbb{N}_0$ is defined as:
> $$\tau_H(m) = \max |H_C| : |C|=m, \; C \subset X$$

> **Lemma (Sauer's Lemma — Sauer, Shelah, Perles).** Let $H$ be a class of functions from $X \to \{0,1\}$ with $\text{VC}(H) = d < \infty$. Then:
> $$\tau_H(m) = 2^m \quad \text{if } m \leq d$$
> $$\tau_H(m) = \left(\frac{e \cdot m}{d}\right)^d \quad \text{if } m > d \quad \text{(VC-Dimension bound)}$$

For any set $C$ of size $m$ and $|H| < \infty$ finite, we know $|H_C| \leq |H|$. So if $|H| < 2^m$, then $H$ cannot shatter $C$ of size $m$. This implies:

$$\text{VC}(H) \leq \log_2(|H|)$$

> **Theorem.** Let $H$ be a class of functions from $X \to \{0,1\}$ and $l: H \times X \times Y \to [0,c]$, $c>0$, a loss function. For any distribution $D$ over $X \times Y$ and $\delta \in (0,1)$, we have with probability at least $1-\delta$ (over the choice of $S \sim D^m$):
> $$\forall h \in H: |L_D(h)-L_S(h)| \leq \sqrt{\frac{8 \cdot \log\!\left(\tau_H(2m) \cdot \frac{4}{\delta}\right)}{m}}$$

This means that as the sample set gets larger, the generalization error and the empirical error come closer together.

---

## Appendix: Preliminaries

### A.1 Sigma-Algebra

Let $S$ be a non-empty set. A family of sets $\mathcal{F} \subset \mathcal{P}(S)$ is called a $\sigma$-Algebra on $S$ if the following conditions hold:

I. $S \in \mathcal{F}$  
II. From $A \in \mathcal{F}$, it follows that $A^C = S \setminus A \in \mathcal{F}$  
III. If $A_i \in \mathcal{F},\, i \in \mathbb{N}$, it follows that $\bigcup_{i=1}^\infty A_i \in \mathcal{F}$

**Remark.** Any subset $\mathcal{F} \subset \mathcal{P}(S)$ is called a family of sets. Smallest $\sigma$-Algebra over $S$: $\{\emptyset, S\}$. The $\sigma$-Algebra generated by $A$: if $A \subset S$, then $\sigma(A) = \{\emptyset, A, A^C, S\}$.

### A.2 Generators

Let $\varepsilon \subset \mathcal{P}(S)$ be a family of sets. Further, let $\Sigma$ be the set of all $\sigma$-Algebras over $S$ that contain $\varepsilon$. Then:

$$\sigma(\varepsilon) = \bigcap_{\mathcal{F} \in \Sigma} \mathcal{F}$$

is called the $\sigma$-Algebra generated by $\varepsilon$.

**Examples.**

$$\varepsilon = \big\{\{1\}\big\}, \quad S=\{1,2,3\}$$
$$\sigma(\varepsilon) = \big\{\emptyset, \{1\}, \{2,3\}, \{1,2,3\}\big\}$$

$$\varepsilon = \big\{\{a\}, \{b\}\big\}, \quad S=\{a,b,c,d\}$$
$$\sigma(\varepsilon) = \big\{\emptyset, S, \{a\}, \{b\}, \{a,b\}, \{b,c,d\}, \{a,c,d\}, \{c,d\}\big\}$$

### A.3 Topological Space

A topological space is a tuple $(X, \tau)$, where $X$ is a set and $\tau$ a collection of subsets of $X$, such that:

I. $\emptyset, X \in \tau$  
II. Closed under union: $\{u_i\}_{i \in I} \subseteq \tau \Rightarrow \bigcup_{i \in I} u_i \in \tau$  
III. Closed under finite intersection: $\{u_i\}_{i=1}^n \subseteq \tau \Rightarrow \bigcap_{i=1}^n u_i \in \tau$ (elements of $\tau$ are called open sets)

**Example.** If $X = \{1,2,3,4\}$, then $\tau = \big\{\emptyset, \{1,2,3,4\}, \{2\}, \{1,2\}, \{2,3\}, \{1,2,3\}\big\}$.

### A.4 Borel-Sigma-Algebra

Let $S$ be a topological space and $\tau$ the system of open subsets of $S$. Then $\mathcal{B}(S) = \sigma(\tau)$ is called the Borel-$\sigma$-Algebra over $S$. Elements $A \in \mathcal{B}(S)$ are called Borel sets. For $S = \mathbb{R}^n$, we write $\mathcal{B}^n = \mathcal{B}(\mathbb{R}^n)$.

Each of the following families of sets is a generator for $\mathcal{B}^n$:

- $\{U \subset \mathbb{R}^n : U \text{ open}\}$
- $\{A \subset \mathbb{R}^n : A \text{ closed}\}$
- $\{]a,b] : a,b \in \mathbb{R}^n, a \leq b\}$ (coordinatewise)
- $\{]-\infty, c] : c \in \mathbb{R}^n\}$, i.e. $]-\infty, c_1] \times ]-\infty, c_2] \times \cdots \times ]-\infty, c_n] \subset \mathbb{R}^n$, with $c=(c_1, \ldots, c_n) \in \mathbb{R}^n$

### A.5 Measurable Space

If $\mathcal{F}$ is a $\sigma$-Algebra over $S$, we call $(S, \mathcal{F})$ a *measurable space*.

**Example.** $\big(\mathbb{R}^n, \mathcal{B}(\mathbb{R}^n)\big)$

### A.6 Measurable Map

Given $(S_1, \mathcal{F}_1)$ and $(S_2, \mathcal{F}_2)$ measurable spaces, we call $f: S_1 \to S_2$ an $(\mathcal{F}_1, \mathcal{F}_2)$-measurable map if $\forall E \in \mathcal{F}_2: f^{-1}(E) \in \mathcal{F}_1$. If $\mathcal{F}_2 = \sigma(\varepsilon)$ where $\varepsilon$ is the generator, then $f$ is measurable if $\forall E \in \varepsilon: f^{-1}(E) \in \mathcal{F}_1$ (we only need to check generators).

**Remark.** If $f$ is continuous, then $f$ is measurable.

### A.7 Measure

Let $(S, \mathcal{F})$ be a measurable space. A function $\mu: \mathcal{F} \to \mathbb{R} \cup \{+\infty, -\infty\} = \bar{\mathbb{R}}$ is called a *measure* if:

I. $\mu(\emptyset) = 0$  
II. $\mu(A) \geq 0$ for all $A \in \mathcal{F}$  
III. For every sequence $(A_n)_{n \in \mathbb{N}}$ of disjoint sets from $\mathcal{F}$:
$$\mu\!\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty \mu(A_i) \qquad \text{($\sigma$-Additivity)}$$

**Example (Counting measure).**
$$A \mapsto \begin{cases} |A| & \text{if } A \text{ finite} \\ \infty & \text{else}\end{cases}$$

### A.8 Measure Space

If $(S, \mathcal{F})$ is a measurable space and $\mu: \mathcal{F} \to \bar{\mathbb{R}}$ a measure, we call $(S, \mathcal{F}, \mu)$ a *measure space*.

Some elementary properties: Let $(S, \mathcal{F}, \mu)$ be a measure space and $A, B, A_n \in \mathcal{F}$, $n \in \mathbb{N}$. Then:

I. If $A$ and $B$ are disjoint: $\mu(A \cup B) = \mu(A) + \mu(B)$  
II. If $A \subset B$ and $\mu(A) < \infty$: $\mu(B \setminus A) = \mu(B) - \mu(A)$  
III. If $A \subset B$: $\mu(A) \leq \mu(B)$  
IV. Sub-$\sigma$-Additivity: $\mu\!\left(\bigcup_{n=1}^\infty A_n\right) \leq \sum_{n=1}^\infty \mu(A_n)$

### A.9 Probability Space

Given that $(S, \mathcal{F}, \mathcal{D})$ is a measure space and $\mathcal{D}(S)=1$, then we call $\mathcal{D}$ a *probability measure* and $(S, \mathcal{F}, \mathcal{D})$ a *probability space*.

- $S$ … the set of *outcomes* of a random experiment.
- $\mathcal{F}$ … the set of *events* to which we want to assign a probability.
- $\mathcal{D}$ … assigns to each event $A \in \mathcal{F}$ a probability $\mathcal{D}(A)$, where $\mathcal{D}: \mathcal{F} \to [0,1]$.

**Remark.** A possibility to construct, based on a measure and a measurable function, a new measure. The *push-forward* measure $\mu_f$ of $\mu$ under $f: S_1 \to S_2$ uses the structure of the source measure space $(S_1, \mathcal{F}_1, \mu)$ and target measurable space $(S_2, \mathcal{F}_2)$.

### A.10 Random Variable

If $(S_1, \mathcal{F}_1, \mathcal{D})$ is a probability space and $(S_2, \mathcal{F}_2)$ a measurable space, then a $(\mathcal{F}_1$-$\mathcal{F}_2)$-measurable function $X: S_1 \to S_2$ is called a *random variable*. If $S_2 = \mathbb{R}^n$, we say $X$ is an $n$-dimensional real random variable.

The push-forward measure $\mathbb{P}_X$ on $S_2$ is also a probability measure, since:

$$\mathbb{P}_X(S_2) = \mathcal{D}(X^{-1}(S_2)) = \mathcal{D}(S_1) = 1$$

We call $\mathbb{P}_X$ the *distribution* of $X$.

**Conventions:**

- $\{X \in A\} = \{w \in S: X(w) \in A\}$
- $\{X = c\} = \{w \in S: X(w) = c\}$
- $\mathbb{P}_X(A) = \mathcal{D}(\{w \in S: X(w) \in A\})$
- $\mathbb{P}_X(\{c\}) = \mathcal{D}(\{w \in S: X(w) = c\})$

#### A.10.1 Expectation

**Motivation: Lebesgue Integration.** The Riemann integral $f: \mathbb{R} \to \mathbb{R}$ has limitations — it cannot be extended to higher dimensions and relies on continuity.

The *Lebesgue integration idea*: instead of partitioning the domain of $f$, we partition the range. We need a way to measure the preimage sets $A_i$; if we can do that, we write $c_i \cdot \mu(A_i)$, where $\mu$ is the measure. This allows us to write:

$$\sum_i c_i \cdot \mu(A_i) \longrightarrow \int_S f \, d\mu$$

**Lebesgue integration for simple functions.** Let $(S, \mathcal{F}, \mu)$ be a measure space with $\mu: \mathcal{F} \to [0,\infty]$. For the indicator function $\mathbb{1}_A: S \to \mathbb{R}$, $A \in \mathcal{F}$:

$$I(\mathbb{1}_A) = \mu(A)$$

A *simple function* is:

$$f(x) = \sum^n_{i=1} c_i \cdot \mathbb{1}_{A_i}(x), \quad A_1, \ldots, A_n \in \mathcal{F}, \quad c_1, \ldots, c_n \in \mathbb{R}$$

To avoid issues with $\infty - \infty$, we restrict to $T^+$, the set of non-negative simple functions. For $f \in T^+$, the Lebesgue integral with respect to measure $\mu$ is:

$$I(f) = \int_S f \, d\mu = \sum_{i=1}^n c_i \cdot \mu(A_i), \quad c_i \geq 0$$

**Monotonicity property:** If $f$ and $g$ are Lebesgue integrable and $f(x) \leq g(x)$, then $\int f \, d\mu \leq \int g \, d\mu$.

> **Definition (Expected Value).** Given a probability space $(S, \mathcal{F}, D)$ and a (quasi-integrable) random variable $X: S \to \mathbb{R}$, we call $\mathbb{E}[X] = \int_S X \, dD$ the *expected value* of $X$.
>
> **Properties:**
> 1. If $X \geq 0$, then $\mathbb{E}[X] \geq 0$
> 2. $\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
> 3. $\mathbb{E}[aX] = a\,\mathbb{E}[X]$
> 4. If $X \leq Y$ (almost everywhere), then $\mathbb{E}[X] \leq \mathbb{E}[Y]$ (given the expectations exist)

#### A.10.2 Markov Inequality

Given a probability space $(S, \mathcal{F}, D)$ and a non-negative random variable $X$, for any $a > 0$:

$$\mathbb{P}[X \geq a] \leq \frac{\mathbb{E}[X]}{a}$$

**Proof.** Define $\varphi: S \to \mathbb{R}$ as:

$$\varphi(x) = \begin{cases} a & \text{if } X \geq a \\ 0 & \text{if } X < a \end{cases}$$

Then $0 \leq \varphi(x) \leq X(x)$. By monotonicity:

$$\int_S X \, dD \geq \int_S \varphi \, dD = a \cdot D(\{x \in S: X(x) \geq a\})$$

Since $a > 0$, dividing by $a$ gives $\mathbb{P}[X \geq a] \leq \frac{\mathbb{E}[X]}{a}$. $\square$

---

## Additional Notes

**Empirical Risk Minimization (ERM).** As a learner only has access to the training data $S$, it is natural to try to select $h$ (our hypothesis) such that the empirical risk is minimized. We call such an $h$ an *empirical risk minimizer*:

$$\mathbb{E}[L_S(h)] = L_{D,f}(h)$$

**Example of a problematic case (overfitting):** Assume the area of the domain (square) is 2 and the area of the inner square is 1. An ERM algorithm returns $h_S$ such that:

$$h_S(x) = \begin{cases} y_i & \text{if } \exists\, i \in \{1, \ldots, m\}: x_i = x \\ 0 & \text{else} \end{cases}$$

(a look-up table). Obviously $L_S(h_S) = 0$, but on unseen instances from $D$ (uniform on $X$), $h_S$ is only correct 50% of the time: $L_{D,f}(h_S) = \frac{1}{2}$. This is *overfitting*.

**Hypothesis class** $H$ (p. 8): We restrict search to $H$ and write:

$$\text{ERM}_H(S) \in \underset{h \in H}{\arg\min} \, L_S(H)$$

**ERM over finite hypothesis class ($|H| < \infty$).** Assumption (realizability): $\exists\, h^* \in H$ with $L_{D,f}(h^*)=0$.

We define:
$$H_{\text{BAD}} = \{h \in H: L_{D,f}(h) > \varepsilon\}$$
$$M = \{S|_X: \exists\, h \in H_{\text{BAD}},\, L_S(h)=0\}$$

The probability that an ERM hypothesis $h_S$ has generalization error $\geq \varepsilon$ is upper bounded by $|H| \cdot e^{-m\varepsilon}$.

> **Corollary.** Let $|H| < \infty$ and $\varepsilon, \delta \in (0,1)$. Further, let $m > \varepsilon^{-1} \log(|H|\delta^{-1})$. Then, for any labeling function $f$ and distribution $D$ (for which realizability holds), with probability at least $1-\delta$ over the choice of $S|_x$ of size $m$, every ERM hypothesis $h_S$ satisfies $L_{D,f}(h_S) \leq \varepsilon$.
>
> *Interpretation:* For sufficiently large $m$, $\text{ERM}_H$ returns a hypothesis $h_S$ that is **P**robably ($1-\delta$) **A**pproximately ($\varepsilon$) **C**orrect (PAC).

> **Definition (PAC Learnability).** A hypothesis class $H$ is *PAC learnable* if there exists a function $m_H: (0,1)^2 \to \mathbb{N}$ and a learning algorithm $A$ such that: for every $\varepsilon, \delta \in (0,1)$, every distribution $D$ over domain $X$, and every labeling function $f: X \to \{0,1\}$, if realizability holds (with respect to $H, D, f$), then running $A$ on $m \geq m_H(\varepsilon, \delta)$ i.i.d. instances from $D$ labeled by $f$ returns a hypothesis $h$ such that with probability at least $1-\delta$:
> $$L_{D,f}(h) \leq \varepsilon$$
>
> $m_H: (0,1)^2 \to \mathbb{N}$ is called the *sample complexity function*.

**Agnostic Setting.** We now remove the realizability assumption — there is no longer an $h^* \in H$ with $L_{D,f}(h^*)=0$. The best we can hope for is a guarantee relative to $\min_{h \in H} L_{D,f}(h)$.

> **Definition (Hoeffding's Inequality).** Let $X_1, \ldots, X_m$ be i.i.d. random variables taking values in $[a_i, b_i]$. Then, with $S_m = \sum_{i=1}^m X_i$:
>
> a) $\mathbb{P}[S_m - \mathbb{E}[S_m] > \varepsilon] \leq \exp\!\left(\frac{-2\varepsilon^2}{\sum_i (b_i - a_i)^2}\right)$
>
> b) $\mathbb{P}[S_m - \mathbb{E}[S_m] < -\varepsilon] \leq \exp\!\left(\frac{-2\varepsilon^2}{\sum_i (b_i - a_i)^2}\right)$
>
> Combined (union bound):
> $$\mathbb{P}[|S_m - \mathbb{E}[S_m]| > \varepsilon] \leq \exp\!\left(\frac{-2\varepsilon^2}{\sum_i (b_i - a_i)^2}\right)$$
>
> Alternative form:
> $$\mathbb{P}\!\left[\left|\frac{1}{m}\sum_{i=1}^m X_i - \mu\right| > \varepsilon\right] \leq \exp\!\left(\frac{-2\varepsilon^2}{(b-a)^2}\right)$$
> with $\mu = \mathbb{E}[X_i]$ and $\mathbb{P}[a \leq X_i \leq b]=1$ for all $i$.

For a fixed $h: X \to Y$ (with $a=0$, $b=1$):

$$\underset{S|_x \sim D^m}{\mathbb{P}}[|L_S(h) - L_{D,f}(h)| > \varepsilon] \leq 2e^{-2\varepsilon^2 m}$$

**Note:** This only holds for a single $h$! For a uniform bound over all $h \in H$ (with $|H| < \infty$, by union bound):

$$\underset{S|_x \sim D^m}{\mathbb{P}}\left[\forall h \in H: |L_S(h) - L_{D,f}(h)| > \varepsilon\right] \leq |H| \cdot 2e^{-2\varepsilon^2 m}$$

Note: realizability was not needed here, but we pay the price of $\varepsilon^2$ vs. $\varepsilon$.

We generalize by letting $D$ be a distribution over $X \times Y$. Adjusted definitions:

$$L_D(h) = \underset{(x,y) \sim D}{\mathbb{P}}[h(x) \neq y] = D\big(\{(x,y) \in X \times Y: h(x) \neq y\}\big)$$
$$L_S(h) = \frac{1}{m} \cdot \bigl|\{i \in \{1, \ldots, m\}: h(x_i) \neq y_i\}\bigr|$$
