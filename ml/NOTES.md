# Notes (phase 1)

## File-name patterns (listed 2026-10-21 via remote_zip.py)
- **LSC50** `VIDEOS.zip`: `VIDEOS/{COLOR_BODY,COLOR_FACE,GRAY_DEPTH,GRAY_IR}/WWWW_PPPP_RRRR.avi`. 50 words (0000-0049) x 5 middle ids (0000-0004) x 4 last ids (0000-0003) = 1000 per folder. Middle = person, last = rep: INFERRED, to confirm by eye (check 1).
- **LSC-54** `Videos Courtesy.zip`: `Videos Courtesy/<signer>/<category>/<word>/<n>.avi`. Categories: Cordialidad, Numeros (accent is mangled in the zip). 43 word folders. 18 signers (1,3,17,19,21-34). 1228 videos in this zip (not ~1500), mostly reps 0-2. Folder names are mangled by encoding (Ba?o, N?meros): match on ASCII only.
- **LSC70** `LSC70/LSC70AN/PerNN/<class>/PerNN_<class>_<k>.jpg` (k=0..5), also `LSC70ANH` (crops) and `LSC70W`. 70 signers x 37 classes: letters A-Z, `NN`(=Ñ), digits 1,4-10, `MIL`, `MILLON`. 15,504 images in LSC70AN (420 per class; MIL/MILLON 402). Note: no digits 2,3 as classes. HandLandmarker found a hand in 39/40 random A-E frames.

Manifest (2026-10-21): 17,102 rows = lsc70 15,504 + lsc50 370 + lsc54 1,228.

## Checks by eye (done 2026-10-21, user confirmed "everything checks out")
- [x] LSC50 middle field = person (5 different people per HOLA); last field = rep.
- [x] LSC-54 Chao == LSC50 ADIOS (0031): same sign, they merge.
- [x] LSC70 frames _0.._5 are in time order.

# Notes (phase 2)

Extraction ran 2026-10-21 on the Colab VM: 4,182 clips (lsc70 2,584 · lsc54 1,228 · lsc50 370). Parts in Drive `soman-ml/frames/<dataset>/`, QC tables in `soman-ml/qc/`.

- **Gate 1 (LSC50 detection):** 0.968 mean, 368/370 clips kept. Passes; no crop or Holistic fallback needed.
- **Gate 1b (LSC70 static letters):** none below 0.75. Passes; `LSC70ANH` fallback not needed. LSC70 overall 0.975, 2,554/2,584 kept.
- **LSC-54:** 1,194/1,228 kept (mean per-frame rate 0.60 because of idle frames at clip edges; the drop rule uses the first-to-last-detection window).
- **Static letters:** every letter has 402-420 frames and 67-70 signers (need >= 250 / >= 50).
- **Words (kept clips / signers):** ADIOS 74/23, BIEN 74/23, BUENAS_NOCHES 77/23, BUENAS_TARDES 75/23, BUENOS_DIAS 74/23, HOLA 70/23, POR_FAVOR 70/22, GRACIAS 46/14 (need >= 9). All pass.
- The LSC-54 digit folders (cinco ... uno) have one signer each; irrelevant to the MVP.
- Gotcha: a `SOMAN_OUT` set in the notebook kernel is inherited by `Popen` children. The full run therefore wrote to `/content/frames_test`, then was copied to Drive.
- Gotcha: never block the colab-mcp kernel with a wait loop; poll with short cells instead.
- Still open: contact sheets looked at by eye; 20-clip LSC50 Holistic cross-check.

# Notes (phase 3)

Built 2026-10-21 (`normalize.py`, `sequences.py`, `build_features.py`, `tests/`; 4 tests pass). Outputs in Drive `soman-ml/features/`.

- `letters.npz`: 6,951 frames after the frame filter (8,313 before; 1,362 dropped = 16%, per the 2x-median rule). 334-361 frames per letter, 70 signers. 8 outliers flagged (4 std), not dropped, still to look at.
- `sequences_words.npz` (T=30, X [1562,30,42], plus X2 for the second hand and locB for variant B) and `sequences_moving.npz` (T=12, 406 of 420 clips). Words file includes impostors (`imp_*`) and all LSC-54 words. Two files instead of one because T differs.
- Idle threshold: 1.51 palm units/s = 3x median wrist speed in the first/last 10% of LSC50 MVP frames (a 95th percentile gave 15.8, which is hands entering the frame, not rest). `features/idle_thr.json`. Moving letters (6 frames) are not trimmed.
- Two-handed (data-driven, second hand >40% of dominant path in most clips) among MVP words: BUENOS_DIAS, BUENAS_TARDES, BUENAS_NOCHES, GRACIAS. HOLA, ADIOS, BIEN, POR_FAVOR are one-handed. Lists in `features/two_handed_words.json`. Check by eye on the contact sheets.
- Front-camera mirroring: contract step 2 (mirror by handshedness label) makes it irrelevant for handshape; Phase 4 golden test must prove it on a phone-mirrored frame.
- Gotcha: `SOMAN_OUT` must point at the frames dir that actually holds the parts (`/content/frames_test` on the current VM; Drive copy is the same).

# Notes (phase 4)

Static-letter MLP trained and exported 2026-10-21 (`train_letters.py`; outputs `ml/out/letters_mlp.json`, `golden/letters.json`, `letters_report.md`, `letters_confusion.png`).

- **Full LSC70 frames do not work for letters.** Hands are ~40 px in full-body 640x480 shots: signer-split accuracy was ~0.69 for every feature set tried (xy, +z, 3D distances; LR and MLP), and a random split only 0.78. Fix = Gate 1b fallback: `LSC70ANH` crops padded 120->240 (`python extract.py lsc70h_static`, dataset `lsc70h`). Result: kNN 0.877, logreg 0.899, MLP+aug 0.917. `build_features.py` now reads `lsc70h` for letters (`SOMAN_LETTER_DS`). **Implication for the app:** the phone must crop around the hand before running HandLandmarker, or hold the hand close to the camera. Normal selfie framing will not match training.
- Gate 2 passes: 14 of 20 letters >= 90%. Chosen (12): A B D I K M Q R U V W X; final held-out: 0.905-0.994 each, none 0.912. Weak letters: E (0.80), C (0.87, confused with E), F/L, P->none.
- tau = 0.859 (5% of none frames above), holdFrames 8. "none" = 2,000 mid-sign frames from LSC50/LSC-54 word clips plus 600 frames of the dropped letters.
- Golden: 20 cases (4 phone-mirrored versions with flipped handedness; features identical). Numpy forward pass of the JSON matches sklearn to 2e-16.
- z experiment skipped (no gain in the diagnostic runs). Augmentation applied to training folds only.
- Colab gotchas: sklearn 1.8 early stopping fails with string labels (encode as ints); `files.download` did not deliver, so outputs were moved as base64 through the cell output.

# Notes (phase 4 v2, 2026-10-22)

Retrained after the webcam demo: the 8 dropped letters (C E F L O P T Y) never appeared because they were absent AND trained into "none". v2 = all 20 static letters + none (none = word-clip frames only). `train_letters.py` compares variants on held-out signers (GroupKFold 5 by signer), macro accuracy over 21 classes:

| variant | frame | sample |
|---|---|---|
| xy, MLP (128,64) (= v1 setup) | 0.918 | 0.928 |
| + z / + geometry (tip distances, joint bends) / both | 0.921 / 0.922 / 0.922 | |
| wider (256,128), alpha 1e-3 | 0.923 | 0.931 |
| aug x4, rotation 25 deg | 0.923 | |
| aug 3D viewpoint (yaw/pitch +-15 deg) | 0.926 | 0.938 |
| + full-frame LSC70 as extra training data | 0.912 (hurts) | |
| **chosen: xy, (256,128), 3D aug, 5-model ensemble** | **0.930** | **0.938** |

- Feature contract unchanged in practice: features = `xy` (same 40 dims as v1 `handshape`). z and geometry gave < 1 point, not worth the Dart code. `normalize.hand_points` + `letter_features` added (xy part == handshape).
- `letters_mlp.json` is format v2: `members` (5 MLPs, average their softmax), `thresholds` per letter, `threshold` free mode (0.70), `holdFrames` 8. Weights rounded to 6 decimals (2.4 MB).
- Target mode (app knows the letter): per-letter threshold = lowest value in [0.50, 0.95] where every other class has <= 5% of frames above it. Most letters 0.50; C 0.58, L 0.77, E 0.78, U 0.95. Weakest target recall: E 0.74, U 0.87, L 0.87, C 0.86, Y 0.90.
- Remaining confusions: P->none 8%, C<->E 6-8%, E->none 6%, T->none 6%, F<->L 4-6%, Y->U 5%.
- The VM had been reset and `frames/lsc70h` was never on Drive; re-extracted (1,400 clips, 7.5 min) and now copied to Drive `soman-ml/frames/lsc70h`. Run log + all trials: Drive `soman-ml/letters_runs.json`, `letters_v2_log.txt`.
- Gotcha: `pkill -f train_letters.py` in a `!` line kills the cell's own shell (the pattern is in its command line).

# Notes (numbers, 2026-10-22)

Same process as letters v2, run with `python extract.py lsc70h_numbers` (694 LSC70ANH crop clips, 3 min) then `python train_letters.py --set numbers`. Outputs: `ml/out/numbers_mlp.json` (same format v2 as letters), `golden/numbers.json`, `numbers_report.md`, `numbers_confusion.png`. Runs log: Drive `soman-ml/numbers_runs.json`, `numbers_log.txt`.

- Classes: 1 4 5 6 7 8 9 10 MIL MILLON + none (LSC70 has no 0, 2, 3). 3,487 frames after the frame filter, 70 signers. none = the same 2,000 word-clip frames (letters are NOT in none).
- Held-out macro (11 classes): xy 0.861; +z/+geo all worse; wider worse; aug x4 0.858; **3D aug 0.868 (chosen, sample 0.908)**; +full-frame LSC70 0.869 and ensemble x5 0.872 below the keep rule. So the model is 1 MLP (128,64), xy features: no new Dart code.
- Strong: 5, 7, 8, 4 (0.93-0.96). Weak: **1 <-> 6** (15-17% each way; target recall 1: 0.35, 6: 0.65 at thresholds 0.85/0.81), 9 -> 4 (14%), MILLON/MIL/10 -> none (12-19%).
- **10, MIL, MILLON move:** within-sample shape spread 0.6-1.4 vs 0.1-0.4 for the others. The static model still gets 0.81-0.87 on them from 6 frames, but they are better suited to the moving-sign path (DTW, Phase 6) if they fail on the phone.
- Golden: 14 cases (4 mirrored), JSON vs sklearn max error 3.2e-6.
- `test_letters_v2.py` golden check loosened to 1e-5 (weights are rounded to 6 decimals since v2; it failed at 1e-9).

# Notes (phase 5)

Words with kNN-DTW (`dtw.py`, `words_dtw.py`; Colab, 2026-10-21). Signer-held-out 5 folds over 23 signers (LSC50 persons spread one per fold), k=5, band 3 of 30 samples (10%), cost = sum of frame distances / path length.

- **Gate 5 FAILS.** With impostor false-accept <= 10%, no word reaches 80% recall. Per the phase rule the 8 words stay video-only ("Mira y elige") until team templates are added (Phase 6) and the numbers are rechecked.
- **Variants (top-1 over the 8 words, no reject, 40 templates/word):** A hand only 0.72 avg; B shoulder-anchored 0.78; **C 0.81** = B + nose-relative position + second hand (shape x0.3, shoulder-relative position). On BUENOS_DIAS / TARDES / NOCHES (avg top-1) A 0.59, B 0.67, C 0.71, so B beats A by 8 and C by 12 points (>= 5 needed), so the **body anchor is adopted: the app must run PoseLandmarker lite** (shoulders + nose, median over the clip). Chosen: C. Feature layout and weights are in each template file (`layout`).
- **C per word (top-1 -> recall at threshold, FA 9.6% on 208 LSC50 impostors):** GRACIAS 0.78 -> 0.33, BUENOS_DIAS 0.74 -> 0.70, BUENAS_TARDES 0.67 -> 0.63, BUENAS_NOCHES 0.73 -> 0.66, BIEN 0.86 -> 0.58, HOLA 0.86 -> 0.61, ADIOS 0.86 -> 0.77, POR_FAVOR 0.96 -> 0.79. Thresholds were fit on the same out-of-fold scores, so these are slightly optimistic.
- **Cross-dataset (templates from one dataset, test on the other, top-1):** C LSC50->LSC-54 0.56, LSC-54->LSC50 0.58 (B 0.62/0.61, A 0.61/0.54). New camera/person generalizes poorly: this is the honest "new user, new phone" number.
- **Why it stops there:** diffuse confusions among BUENOS/BUENAS_* and GRACIAS; the hand-only location (relative to the first frame) is close to noise (0.26 alone). Swapping hand order, delta features, other k and weights did not move it past ~0.82.
- **Templates:** 40 medoids/word (not 20: in early tests 20 medoids scored 0.74 vs 0.79 for 40/all), rounded to 3 decimals, `ml/out/templates/<word>.json` (5 MB total, DTW over 320 templates is cheap). Each carries `source`, `threshold`, `two_handed`, `layout`, `feature_version` v2. `golden/words.json` holds 20 query cases with expected class scores plus one pure DTW pair; tolerance 1e-4.
- **Impostor limits:** only LSC50 impostors have pose (208 clips from 5 people); LSC-54 non-MVP signs have none. LSC-54 courtesy signs besides the 8 words were not used as extra impostors for the C/B tests (variant A used 1,002).
- **Two-handed (data-driven):** BUENOS_DIAS, BUENAS_TARDES, BUENAS_NOCHES, GRACIAS. CHAO = ADIOS merged (checked in Phase 1).
- Sequences now also hold `locN`, `locB2`, `locN2` (nose-relative / second hand). The pose pass also ran on the LSC50 impostors.
- Gotcha: `pkill -f <name>` inside a `subprocess.run(shell=True)` kills its own shell (the pattern is in its command line).
