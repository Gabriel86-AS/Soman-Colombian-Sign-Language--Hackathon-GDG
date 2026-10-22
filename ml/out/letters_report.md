# Letter model report

Frames: 6959 letters (20 classes, 70 signers) + 2000 none.

## Held-out-signer accuracy (all frames)

- kNN k=5: 0.877
- logreg: 0.899
- MLP (128,64) + aug: 0.917

## Per-letter accuracy (MLP, held-out signers)

- A: 0.946
- B: 0.930
- C: 0.866
- D: 0.928
- E: 0.801
- F: 0.895
- I: 0.941
- K: 0.933
- L: 0.916
- M: 0.929
- O: 0.893
- P: 0.885
- Q: 0.949
- R: 0.930
- T: 0.906
- U: 0.963
- V: 0.946
- W: 0.966
- X: 0.927
- Y: 0.887

## Most confused (true -> predicted)

- P -> none: 0.090
- C -> E: 0.078
- E -> C: 0.070
- F -> L: 0.068
- E -> none: 0.067
- L -> F: 0.067
- T -> none: 0.057
- Y -> U: 0.054
- E -> O: 0.052
- O -> none: 0.051

Letters at >= 90%: 14 of 20. **Gate 2 (>= 8): PASS**
Chosen MVP letters (12): A B D I K M Q R U V W X

## Final set, held-out signers

- A: 0.994
- B: 0.939
- D: 0.934
- I: 0.950
- K: 0.917
- M: 0.960
- Q: 0.962
- R: 0.959
- U: 0.943
- V: 0.968
- W: 0.969
- X: 0.905
- none: 0.912

Threshold tau = 0.859 (<= 5% of none frames above it; floor 0.70). holdFrames = 8.

Golden: 20 cases (4 mirrored); JSON forward pass vs sklearn max error 2.22e-16
