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
