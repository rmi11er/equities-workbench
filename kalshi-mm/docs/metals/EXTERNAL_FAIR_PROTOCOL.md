# External-fair attribution protocol

Status: frozen hypothesis-generation protocol. The existing Kalshi-candle results are
not a fair-value backtest and cannot populate this report without synchronized SI,
Pyth, and Kalshi events.

## Research question

The first question is whether the approximately 2.35-cent penetration loss from fill
to one second is caused by a stale Kalshi quote or by information in Kalshi flow that
remains after contemporaneous futures movement is removed.

For an acquired outcome `A`, entry cost `c`, quote time `t_q`, and fill time `t_f`, use
the identity

\[
Y_A-c = [p_A(t_q)-c] + [p_A(t_f)-p_A(t_q)]
        + [p_A(t_f+1s)-p_A(t_f)] + [Y_A-p_A(t_f+1s)].
\]

The four terms are quoted edge, stale-quote movement, post-fill external movement,
and forecast residual. A one-second Kalshi markout uses the same first three terms
and replaces the settlement residual with `K_A(t_f+1s)-p_A(t_f+1s)`.

All component tables report dollars, cents, day-block confidence intervals, and the
signed percentage of the total loss. The components must reconcile to the reported
total before a run is valid. Model/timestamp uncertainty is reported separately and
is not inserted as an invented balancing term.

## Event-time contract

Every source record retains:

- source/exchange event time;
- vendor capture or publication time when supplied;
- UTC local receive time;
- local monotonic receive time;
- ingestion sequence as the deterministic tie-break;
- connection and native source sequence identifiers when supplied.

Replay and feature joins use `(local_monotonic_receive_ns, ingestion_sequence)` only.
Source, vendor, and publication timestamps are diagnostic fields. Every feature row
must satisfy `max(input_receive_key) <= decision_receive_key`.

The mandatory futures lag sweep is `-500, -250, -100, 0, +100, +250, +500` ms.
Negative shifts deliberately make futures older. Positive shifts are explicit
noncausal controls: results using a futures message received after the relevant
decision are leakage and can never enter policy results. Report the best causal and
best unrestricted shift. If the optimum is noncausal, or attribution is unstable
under a 100--250 ms shift, the clock gate fails.

## SI-to-Pyth benchmark replicator

Start with silver only:

\[
\widehat B_t=B_u+\beta_s(F_t-F_u)+c_s,
\]

where `u` is the most recent received Pyth publication and `s` is a predeclared
session. Fit by expanding or rolling chronological training data. Do not choose
sessions, lag, or thresholds after viewing the final holdout.

Before the estimator may feed trading probabilities, it must pass all of these gates:

1. lower held-out next-Pyth error than the unchanged-Pyth baseline;
2. predeclared near-zero residual-bias limits in every sufficiently sampled session;
3. predeclared beta-stability limits across held-out days;
4. no material residual relationship with signed SI changes;
5. exact reconstruction of every included settlement target and final observation;
6. no feature with a receive key later than its decision key;
7. predeclared state-dependent error limits relative to benchmark/strike distance.

Report MAE, RMSE, bias, session bias, daily/session beta, residual/SI correlation,
coverage, and error-to-strike-distance quantiles. A historical Pyth candle whose
`receive_ts` is merely the later backfill-request time is not a live causal tick and
cannot satisfy the live replay clock contract.

## Frozen residual-flow cuts

Use the current dates only for hypothesis generation. Freeze these cuts before a
later chronological holdout:

- price: 5--15, 15--25, 25--40, 40--60, 60--75, 75--85, 85--95 cents;
- remaining time: 10--15, 5--10, 2--5, 1--2, and less than 1 minute;
- signed SI momentum over 100 ms, 250 ms, 1 s, 5 s, and 30 s;
- favorite age after first crossing 70, 80, and 90 cents;
- newly established versus stable favorite;
- normal versus predeclared scheduled-event windows;
- Asia, London, COMEX open, and US afternoon sessions;
- Kalshi trade imbalance and consecutive taker direction;
- spread, depth, estimated quantity ahead, Pyth confidence, and feed age.

Primary interactions are `(price, remaining time, signed SI momentum, favorite age)`.
In particular, compare newly crossed 80-cent favorites with favorites stable above 80
cents for 30--60 seconds, separating continued SI momentum from stable/reversing SI.

## Attribution decisions

- More than 70% of the one-second loss explained by causally available futures:
  prioritize cancel latency, quote freshness, and feed health.
- 30--70% explained: combine stale-quote protection with a residual-flow model.
- Less than 30% explained: treat Kalshi fills as independently informed and require
  materially wider entry hurdles.
- Attribution unstable under a 100--250 ms shift: fail the clock gate and do not trade.

These are diagnostic classifications, not automatic authorization to trade.

## Policy order

1. SI/Pyth timing and benchmark replication.
2. One-second external-fair attribution.
3. Residual toxicity by price, time, momentum, and favorite age.
4. Fair-gated single-leg replay.
5. State-conditional pair completion and first-leg stop policy.
6. Queue-aware replay.
7. More complex volatility distributions.

Paired quoting remains secondary. Evaluate it after the first fill as

\[
EV_{pair}=P(complete\mid X_{post-fill})L
          -P(fail\mid X_{post-fill})C_{single}.
\]

The unconditional 47.1% penetration completion rate is not a policy input.

## Initial eligibility form

The long-vega policy may quote only if

\[
p_{external}-c > A_{residual}(X)+U(X)+R(q,X)+fees.
\]

This protocol specifies research outputs only. It does not enable authenticated order
submission, live capital, or a production trading state.
