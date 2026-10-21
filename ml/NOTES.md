# Notes (phase 1)

## File-name patterns (listed 2026-10-21 via remote_zip.py)
- **LSC50** `VIDEOS.zip`: `VIDEOS/{COLOR_BODY,COLOR_FACE,GRAY_DEPTH,GRAY_IR}/WWWW_PPPP_RRRR.avi`. 50 words (0000-0049) x 5 middle ids (0000-0004) x 4 last ids (0000-0003) = 1000 per folder. Middle = person, last = rep: INFERRED, to confirm by eye (check 1).
- **LSC-54** `Videos Courtesy.zip`: `Videos Courtesy/<signer>/<category>/<word>/<n>.avi`. Categories: Cordialidad, Numeros (accent is mangled in the zip). 43 word folders. 18 signers (1,3,17,19,21-34). 1228 videos in this zip (not ~1500), mostly reps 0-2. Folder names are mangled by encoding (Ba?o, N?meros): match on ASCII only.
- **LSC70**: pending first download.

## Checks by eye (pending)
- [ ] LSC50 middle field is the person? (watch 0030_* per id)
- [ ] LSC-54 Chao == LSC50 ADIOS (0031)?
- [ ] LSC70 frames _0.._5 in time order?
