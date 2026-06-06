# Backlog

- Run the real experiment on a PopQA slice on a rented GPU.
- Compare branch-disagreement AUROC against the three baselines (DeLong).
- Add TriviaQA as a secondary dataset check.
- Record the AUROC-vs-token-cost frontier and write it up in `experiments/`.
- Add reading notes for semantic entropy and self-consistency under `papers/`.
- Investigate the popularity axis: does the detector fire hardest on obscure
  (low-popularity) PopQA facts?
- Decide whether the localised-marker branch mode beats plain self-consistency as
  a disagreement source at equal token budget.
