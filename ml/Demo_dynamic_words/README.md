# Dynamic words demo (8 MVP words)

Live webcam test of the word model `ml/out/words_mlp_swap.json` (round 4 winner: "swap", both hand orders, pass floor 0.3). It works like the app's "Haz la seña" for words: you pick a word, get a countdown and sign during a 2.5 s window.

## Run

Double-click `run.bat`, or from the `ml/` folder:

```
.venv-demo\Scripts\python Demo_dynamic_words\demo.py                       # webcam 0
.venv-demo\Scripts\python Demo_dynamic_words\demo.py 1                     # another camera
.venv-demo\Scripts\python Demo_dynamic_words\demo.py --check               # no camera: golden test
.venv-demo\Scripts\python Demo_dynamic_words\demo.py --model words_mlp     # compare with the old model
```

The first run downloads `pose_landmarker_lite.task` into `ml/models/`.

## Keys

`1` Gracias, `2` Buenos días, `3` Buenas tardes, `4` Buenas noches, `5` Bien, `6` Hola, `7` Adiós, `8` Por favor. Space = try the same word again, Esc = quit.

## How it works

1. Countdown (3 s), then a 2.5 s recording window. Start and end the sign inside the window.
2. Each frame: HandLandmarker (up to 2 hands) and PoseLandmarker lite (nose and shoulders) on the full frame.
3. `sequences.clip_to_sequence` (same code as training): hand tracks, idle trim, resample to 30 steps; then 15 steps.
4. The clip is scored twice, once with the hands as found and once with the two hands swapped, and the two results are averaged. This fixes the "which hand is the main one" flip between people.
5. Pass = p(word) at or above its red tick. The panel shows one bar per word plus "otra seña" (other signs). `*` = the word passed Gate 5 on held-out signers.

## Tips

- Stand back so the camera sees your head, both shoulders and both hands (the model uses hand position relative to the body).
- Good light, plain background.
- Words without `*` can still be tried; they are just less reliable.
