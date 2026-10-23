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

## Retrain with team captures (2026-10-22)
`python train_letters.py --set numbers --custom --quick`, adding 1,250 frames from 3 team signers (aris_p3, gabriel_p1, santiago_p2; numbers 1-5) as their own signer groups. The exported `numbers_mlp.json` now has 12 numbers (adds 2 and 3, only from our captures) + none, 73 signers, 1 MLP (128,64), xy features.
- Held-out macro 0.809 frame / 0.852 sample. Team captures (each person held out) 0.704; LSC70 + none 0.865.
- Team per sign: 1 0.90, 2 0.71, 3 **0.43**, 4 0.79, 5 0.68. aris is the hardest signer (2/3/5 ~0.3-0.5); santiago the easiest.
- Main confusions: **3 -> 8 (35%)**, 6 -> 1 (21%), 2 -> 7 (14%). More 3s from more people are the next fix.

## Numbers 1-5 only (2026-10-22, current `numbers_mlp.json`)
The intended model is numbers 1-5, so the 12-number one above is replaced. `python train_letters.py --set numbers --custom --only 1,2,3,4,5` (full variant search): LSC70 1/4/5 + team 1-5 + none, 73 signers, 2,249 number frames.
- Chosen: xy, (128,64), aug x4 rot 25 (0.904 vs base 0.883). Held-out macro **0.904 frame / 0.910 sample**; LSC70 + none 0.965; team captures 0.862.
- Per sign (all): 1 0.98, 2 0.89, 3 0.78, 4 0.96, 5 0.87, none 0.95. Team: 1 1.00, 2 0.89, 3 0.78, 4 0.92, 5 0.72 (gabriel's 5 0.49).
- Errors are now mostly "sign -> none" (5 13%, 3 13%, 2 10%), not number swaps; 3 -> 4 6%.
- Demo: `ml/demo_numbers_team/` (frozen copy of this model).

# Notes (phase 6, moving letters, 2026-10-22)

`moving_dtw.py` (sheets / eval / mlp / export). Clips = LSC70's 6 frames (first second, 6 fps) of G H J N Ñ S Z; negatives = the 20 static letters' clips. 1,832 clips kept (458 moving, 1,374 static, 70 signers; 58 dropped for < 4 frames with a hand). Crops: `python extract.py lsc70h_moving` (490 clips; **S was missing from `MOVING`, fixed**). Sheets: Drive `soman-ml/qc/sheets/moving_<X>.png` (6 crops + full-frame wrist path per clip).

- **Shape from the hand crop is the key** (as for static letters): DTW mean recall 0.73 crop vs 0.45 full frame. The full-frame wrist path **hurts** (0.73 -> 0.68 at weight 1, 0.54 at weight 6): hands are ~40 px, the tracker jumps. So: no location features for moving letters.
- **DTW verifies handshape, not motion.** With the words threshold rule (cap 95th pct, <= 10% of ALL non-target clips) DTW crop_b2 passes G 0.87 / N 0.93, but a held K passes as G 75%, M as N 60%, R as H 70%, F as S 74%. With a fair rule (every other single letter <= 5%, the letters rule), DTW drops to 0.36 mean recall (0.65 with static reject templates), 0/7 letters pass.
- **MLP wins (M9 rule):** sklearn MLP (256,128) on the crop-shape sequence [12x40 = 480], classes 7 moving + "static", aug x4 (in-plane rot 15, scale 10%, jitter 0.02), 3 seeds, GroupKFold 5 by signer. worst5 rule, p >= 0.5: mean recall 0.79, **G 0.87, J 0.84, N 0.96, Ñ 0.86 pass**; S 0.66, Z 0.73, H 0.62 fail; worst static letter passing <= 3%. Without aug 0.70.
- Sheets: motion is mostly inside the first second (J pinky turn, Z index trace visible) but frames are blurry.
- **User chose the MLP.** `train_moving.py` variant search (mean recall, worst5 rule): base 0.797, +3D aug 0.814, **+time warp 0.838 (chosen)**, copies 8 0.833, rot 25 0.838 (no gain), small (128,64) 0.819, alpha 1e-2 0.832, 5 seeds 0.820. Final: (256,128), alpha 1e-3, aug x4 rot 15 + 3D yaw/pitch 15 + time warp 0.85-1.15x / shift 10%, 3 members.
- **Final (held-out signers):** G 0.89, J 0.88, N 0.96, Ñ 0.90, Z 0.80 pass Gate 3 (numbers); H 0.67, S 0.76 fail (video-only). Worst static letter passing as a moving one <= 3%. Thresholds 0.50 except S 0.61, Z 0.52.
- Exported `ml/out/moving_mlp.json` (format v2 + `kind`, `features` moving_shape_m1, `T` 12, `frames` 6, `window_ms` 1000, `gate3`; 4.4 MB, 3 members), `golden/moving_letters.json` (18 cases with raw crop landmarks, null = no hand; JSON vs sklearn 6e-7), `moving_report.md`. Demo `ml/demo_moving/` (countdown, 1 s, 6 frames at LSC70 timing; `--check` passes locally).
- Gate 3 eye check on the sheets: still to confirm by the user (J and Z looked complete in the first second).
- Gotcha: printing > ~600 KB in one colab-mcp cell output hangs and drops the connection; 200 KB chunks work.

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

# Notes (words MLP, 2026-10-23)

`train_words.py` replaces kNN-DTW for words with an MLP (same move as moving letters). Input = variant-C blocks from `sequences_words.npz`, classes 8 MVP words + other (208 LSC50 impostors), same 5 signer folds as the DTW eval. Two search rounds (`--round2 --base '<json>'`); logs on Drive `soman-ml/features/words_runs*.json`, `soman-ml/words_mlp_r2/`; out-of-fold probs `features/words_oof_mlp*.npy`. Full table in `out/words_mlp_report.md`.

- **4/8 words pass** (BIEN 0.84, HOLA 0.83, ADIOS 0.84, POR_FAVOR 0.89), mean recall 0.77, impostor FA 0.5-4.3% per word. DTW: 0/8, mean 0.63. **Gate 5 still fails** (6 needed). Top-1 equal (0.81). Cross-dataset 0.61 / 0.59 (DTW 0.56 / 0.58).
- Final: T=15 (each block resampled 30 -> 15), no second-hand shape (it hurt BUENAS_NOCHES 0.51 -> 0.69 when dropped), aug x8 (hand rotation 15, position shift 0.2 shoulder widths + scale 10%, time warp), (256,128), 3 members, 6.0 MB.
- What helped: augmentation (+5 points), stronger position shift, dropping shape2, T15. What did not: location velocity, fingertip-only shape2, oversampling GRACIAS, shift 0.3, smaller net, ensemble 5.
- Thresholds all sit at the 0.5 floor: the reject rule is not the limit, model confidence is. A 0.3 floor adds 2-3 points but no extra word passes.
- Weak: GRACIAS 0.54 (46 clips, 14 signers), BUENAS_TARDES 0.69, BUENOS_DIAS / BUENAS_NOCHES 0.77 (confused with each other). More signers (team clips) is the next fix.
- Gotcha: round-1 model and golden share the filename `words_mlp.json`; copy them to Drive under different names.
- Outputs `out/words_mlp.json` (format v2 + `kind` words, `layout`, `T`, `T_source`, `needs_pose`, `gate5`) and `out/golden/words_mlp.json` (20 cases: T=30 blocks, flat features, probs; JSON vs sklearn 7e-8) are in Drive `soman-ml/words_mlp_r2/words_out.zip` (6 MB model, too big for the colab-mcp output channel).

# Notes (words, two hands, 2026-10-23)

Rounds 3 and 4 of `train_words.py` (`--round3`, `--round4`, logs `features/words_runs_r3.json`, `_r4.json`). Round 4 = each setup x 3 repeats with other random draws (run-to-run sd ~0.005, but single runs of the same config moved 0.746 -> 0.770 when the random stream changed).

- **Hands swap.** "Main = longest wrist path" flips in two-handed words: second/main path ratio > 0.7 in 23-36% of GRACIAS / BUENOS / BUENAS clips; the second hand is the higher one in only 48-68%. This is why second-hand shape hurt before.
- Round 4 means (recall @0.5 floor, words pass @0.5, words pass @0.3 floor, top-1): base 0.756 / 3.7 / 3.7 / 0.806; **swap** (train on both hand orders, average the two orders at test) 0.749 / 3.0 / **5.7** / **0.838**; swap+shape2 0.775 / 3.7 / 4.3 (GRACIAS 0.71 vs 0.52); canon_shape2 (main = higher hand) 0.771 / 4.7 / 5.0; swap+all (shape2, locN2, hand offset) 0.763 / 4.0 / 4.0; canon_all 0.774 / 4.0 / 5.0.
- Second-hand nose position and hand-to-hand offset alone: +0.01, within noise.
- Picked: **swap at floor 0.3**. Final export run: **6/8 pass, Gate 5 PASS** (GRACIAS 0.67 and BUENAS_NOCHES 0.68 fail), mean recall 0.823, impostor FA 0-8%; cross-dataset 0.50 / 0.58 (worse than the old 0.61 / 0.59). Exported as `out/words_mlp_swap.json` (+ golden, report; `hand_order` "both", `idle_thr`, `floor` in the JSON). Old `words_mlp.json` kept. Demo: `ml/Demo_dynamic_words/`.
