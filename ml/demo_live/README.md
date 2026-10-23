# Live fingerspelling demo (always on)

The camera watches all the time, with no countdown and no target letter. Every letter it recognizes is typed into a line of text at the bottom.

- **Static letters** (A B C D E F I K L M O P Q R T U V W X Y), from `ml/out/letters_mlp.json`.
- **Moving letters** G J N Ñ Z, from `ml/out/moving_mlp.json`. H and S failed their accuracy check; add `--all-moving` to try them anyway.

## Run

Double-click `run.bat`, or from the `ml/` folder:

```
.venv-demo\Scripts\python demo_live\demo.py            # webcam 0
.venv-demo\Scripts\python demo_live\demo.py 1          # another camera
.venv-demo\Scripts\python demo_live\demo.py --all-moving
```

## Keys

Space = add a space, Backspace = delete the last letter, `0` or Delete = clear, Esc = quit.

## How it decides

- **Static letter:** the top guess over its threshold for 8 frames in a row (green bar under the status line) **and** the hand is steady (motion below 0.35 over the last 0.4 s). The same letter types again only after the hand shows something else for a moment.
- **Moving letter:** every frame, the last 1 s of video is checked (6 frames at LSC70's timing, like training). It types when the same moving letter wins with p >= 0.80 for 3 frames in a row (orange bar). The model has a "static" class, so a held hand does not count as a moving letter.
- After any letter: 0.8 s pause before the next one.

## Screen

- Left: camera, status (steady / moving / cooldown) with the motion value, the typed text.
- Right: the hand crop, the top static-letter guesses, the top moving-letter guesses for the last second.

## Tips

- Hold the hand close to the camera (both models were trained on hand crops).
- Do moving letters in one smooth motion of about 1 second. Drop or relax the hand afterwards so the final pose is not typed as a static letter.
- Tuning values (thresholds, hold frames, cooldown) are at the top of `demo.py`.
