# G4d Oracle-Gap Report

Verdict: **NULL/NEGATIVE**.
G5 selector status: **BLOCKED**.
Because the G4d result is null/negative, G5 remains blocked.

## Scope

G4d ran paired complete-mode comparisons only: Arm A was FloodExpertStateMachine and Arm B was ControlledRetransmissionExpert. Each arm used the same scenario config, seed, mobility plan, traffic schedule, and initialization conditions. Runtime byte identity after policy divergence was not required.

Seeds: 1 through 10 (n=10 per scenario).

Thresholds: A-042 epsilon_pdr=0.02, A-043 epsilon_tx_reduction=0.05, A-044 epsilon_gap=0.05. A-045 is reserved and inactive; A-046 is Plan A only.

The relative tx-attempts delta is `(tx_attempts_controlled - tx_attempts_flood) / max(tx_attempts_flood, 1)`. `tx_attempts` is a proxy, not hardware energy or validated airtime.

For PDR and relative tx-attempts deltas, CIs are paired-seed t-distribution intervals. For oracle_gap_score, the CI is a Clopper-Pearson binomial proportion interval over seeds where the mode-level oracle selects controlled retransmission.

## Per-Scenario CI Summary

| Scenario | PDR flood | PDR controlled | Delta PDR CI | Tx flood | Tx controlled | Rel tx CI | Oracle gap CI | Pairing | Positive |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| easy | 1.0000 | 1.0000 | [0.0000, 0.0000] | 144.00 | 144.00 | [0.0000, 0.0000] | 0.0000 [0.0000, 0.3085] | pass | no |
| p5_baseline_normal | 0.6868 | 0.6887 | [-0.0024, 0.0062] | 359.10 | 379.80 | [-0.0189, 0.0835] | 0.0000 [0.0000, 0.3085] | pass | no |
| p5_congestion_burst | 0.6792 | 0.6786 | [-0.0119, 0.0107] | 1028.50 | 1049.10 | [0.0024, 0.0365] | 0.1000 [0.0025, 0.4450] | pass | no |
| p5_low_power_survival | 0.2396 | 0.2377 | [-0.0062, 0.0024] | 224.90 | 220.40 | [-0.0469, 0.0242] | 0.1000 [0.0025, 0.4450] | pass | no |
| p5_interference_zone | 0.5585 | 0.5755 | [-0.0264, 0.0603] | 1259.80 | 2130.00 | [0.4319, 0.8398] | 0.0000 [0.0000, 0.3085] | pass | no |
| p5_partition_reconnect | 0.5528 | 0.5887 | [0.0185, 0.0532] | 1542.50 | 3832.30 | [1.0869, 1.7397] | 0.0000 [0.0000, 0.3085] | pass | no |
| p5_node_failure | 0.6755 | 0.6755 | [-0.0064, 0.0064] | 307.50 | 328.20 | [-0.0816, 0.2037] | 0.1000 [0.0025, 0.4450] | pass | no |
| p5_group_mobility | 0.6585 | 0.6736 | [-0.0110, 0.0412] | 1053.70 | 1476.80 | [0.1827, 0.4524] | 0.0000 [0.0000, 0.3085] | pass | no |
| p5_priority_emergency | 0.6906 | 0.6881 | [-0.0122, 0.0072] | 1902.80 | 1926.70 | [-0.0205, 0.0552] | 0.2000 [0.0252, 0.5561] | pass | no |

## Same-Scenario Gate

A positive verdict requires one scenario to satisfy all three CI conditions together: PDR safety lower bound, tx-attempt reduction upper bound, and oracle-gap lower bound. Evidence is not mixed across scenarios and point estimates are not used as the final gate.

## Output Tables

- `outputs/g4d_mode_comparison_results.csv`
- `outputs/g4d_oracle_gap_summary.csv`

## Limitations

- Results are within-simulator mode-level counterfactual characterizations.
- Controlled retransmission is the pre-registered immediate controlled-redundancy expert; no tuning was applied for G4d.
- `tx_attempts` is only a cost proxy.
- No selector, ML, RL, DRL, GNN, route-discovery protocol, timer hook, or per-decision intervention artifact is authorized by this report.
- No compliance or hardware-transfer claim is made.
