# Direct Preference Optimization: Derivation and Experimental Findings

This document derives the DPO loss step by step and records what the
experiments in `src/phase5_alignment/` did and did not show.

---

## 1. The Goal: KL-Regularized Reward Maximization

RLHF wants a policy pi that scores well under a reward function r while
staying close to a reference policy pi_ref:

    max over pi:  E_{x, y ~ pi}[ r(x, y) ]  -  beta * KL( pi(y|x) || pi_ref(y|x) )

The KL term prevents the policy from drifting into degenerate outputs that
exploit the reward. beta sets how strong the constraint is.

---

## 2. The Closed-Form Optimal Policy

This objective has a known optimal solution:

    pi*(y|x) = (1 / Z(x)) * pi_ref(y|x) * exp( r(x, y) / beta )

where Z(x) = sum over y of pi_ref(y|x) * exp( r(x, y) / beta ) is a
normalizing constant. Z(x) is intractable to compute because it sums over
all possible sequences.

---

## 3. Solving for the Reward

Take the log of both sides and rearrange:

    r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)

The reward is expressed through the policy itself, up to a term that depends
only on the prompt x.

---

## 4. The Bradley-Terry Preference Model

Given a preferred completion y_w and a dispreferred completion y_l for the
same prompt x, preferences are modeled as:

    P(y_w preferred over y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )

---

## 5. The Cancellation

Substitute the reward from Section 3 into Section 4. Both completions share
the same prompt, so the beta * log Z(x) terms are identical and cancel:

    r(x, y_w) - r(x, y_l)
        = beta * log( pi*(y_w|x) / pi_ref(y_w|x) )
        - beta * log( pi*(y_l|x) / pi_ref(y_l|x) )

This removes the intractable partition function and the need for an explicit
reward model.

---

## 6. The DPO Loss

Replace pi* with the trainable policy pi_theta and maximize the likelihood of
the observed preferences:

    L_DPO(theta) = - E[ log sigmoid( beta * ( log( pi_theta(y_w|x) / pi_ref(y_w|x) )
                                            - log( pi_theta(y_l|x) / pi_ref(y_l|x) ) ) ) ]

Each sequence log-probability is the sum of per-token log-probabilities of the
completion given the prompt (`compute_sequence_logprob` in `dpo_loss.py`).
The reference model is a frozen deep copy taken before training.

**Gradient interpretation.** With margin m = beta * (log-ratio_w - log-ratio_l),
the gradient of the loss is proportional to sigmoid(-m) times the gradient of
(log pi_theta(y_w|x) - log pi_theta(y_l|x)). Pairs the policy currently ranks
wrongly get the largest updates, and pairs it already ranks correctly fade
out. Larger beta rescales m, so the policy reaches a given margin with a
smaller change in log-probabilities, which is why beta acts as the drift dial.

---

## 7. Verification Before Training

From `scripts/test_dpo_loss.py`:

    Log-prob of a real continuation:        -63.86  (2.13 nats/token)
    Log-prob of random nonsense:            -217.87
    Loss with policy == reference:          0.5309  (margin +0.356)
    Total gradient norm through the loss:   31.87    (finite, non-zero)

The margin is already positive for real-vs-random because the base model was
trained on real text. Note that a loss of log(2) = 0.693 would appear if the
margin were exactly zero; 0.5309 corresponds to the positive margin.

---

## 8. Preference Data: Anti-Repetition

For each prompt, 4 completions were sampled at temperature 1.2, scored by
4-gram repetition rate, and the least/most repetitive became winner/loser.
Pairs with a rate gap under 0.05 were dropped.

    100 pairs:  average winner repetition = 0.0000
                average loser repetition  = 0.0646

---

## 9. Training Results (beta = 0.1, 20 epochs, 100 pairs)

    Metric                Epoch 1     Epoch 20
    DPO loss              0.6813      0.2305
    Implicit reward margin 0.0263     3.4106
    Win rate              0.62        0.89
    Approx KL             1.40        37.37

Loss, margin and win rate move as expected. The approximate KL grew about
27x and had not plateaued, so beta = 0.1 did not keep the policy close to
the reference at this learning rate and epoch count.

Free-running generation from "ROMEO:" (one sample each) showed repetition
0.0837 for the base model and 0.0345 for the aligned model, but the aligned
text was visibly more chaotic (mid-word capitalization, garbled names).

---

## 10. Beta Ablation

Same pairs, same checkpoint, same learning rate:

    Beta     Reward Margin    Approx KL    Win Rate    Repetition
    0.05          2.33          60.10        0.90        0.0148
    0.1           3.66          39.39        0.95        0.0049
    0.5           8.42           6.21        0.98        0.0345
    2.0          19.25           3.17        0.97        0.0099

**What is supported:** approximate KL decreases monotonically as beta rises
(60.1 to 3.2), matching the theory that beta anchors the policy to the
reference.

**What is not a quality signal:** reward margin grows with beta because the
DPO logit is beta times a log-ratio difference.

**What is inconclusive:** repetition was not monotonic in beta (best at 0.1,
worse at 0.5 than at 0.05). Each value is one stochastic sample at
temperature 0.8, so differences of this size cannot be distinguished from
sampling noise.

---

## 11. Limitations

- The KL reported is a proxy: mean absolute change in log-probability on
  winner completions, not the exact KL over the output distribution.
- Repetition must be averaged over many generations per beta to support any
  ordering claim.
- Model size (~104K parameters), dataset size (100 pairs) and a single
  learning rate limit conclusions. No tested setting reduced repetition
  while preserving fluency; this is consistent with the alignment tax reported
  in the RLHF/DPO literature, but our experiments do not isolate its cause.
- The preference signal is a heuristic (n-gram repetition), not human
  judgment.

---

## 12. Connection to the Rest of the Capstone

The KL anchor plays the same role as the residual "+x" of Phases 2 and 4 and
the forget gate of Phase 3: something that keeps the current computation tied
to a stable reference so optimization does not run off. That analogy is
conceptual, not a claim these experiments tested.