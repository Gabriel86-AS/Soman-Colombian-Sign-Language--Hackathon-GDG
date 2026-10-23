# Soman · Aprende Lengua de Señas Colombiana

**Soman** es una app Android, al estilo Duolingo, para aprender **Lengua de Señas Colombiana (LSC)**. La cámara del celular reconoce tus señas **en el propio teléfono**, sin enviar video a ningún servidor. Nos acompaña **Nua**, una iguana con capota a la que le encanta el tinto: te da consejos, te anima en el camino de lecciones y celebra contigo.

Proyecto creado para el **Hackathon GDG** (MVP de 1–2 días).

---

## Contenido

1. [Qué hace](#qué-hace)
2. [Estado del proyecto](#estado-del-proyecto)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Tecnologías y versiones](#tecnologías-y-versiones)
5. [Cómo funciona el reconocimiento](#cómo-funciona-el-reconocimiento)
6. [Modelos y resultados](#modelos-y-resultados)
7. [Datasets](#datasets)
8. [Cómo ejecutarlo](#cómo-ejecutarlo)
9. [Diseño](#diseño)
10. [Licencias y créditos](#licencias-y-créditos)

---

## Qué hace

La app tiene tres unidades:

| Unidad | Señas | Ejercicios |
|---|---|---|
| **El abecedario** | Letras estáticas A B C D E F I K L M O P Q R T U V W X Y · letras con movimiento G J N Ñ Z | Mira y elige, Haz la seña (cámara), Deletrea |
| **Números** | 1 2 3 4 5 | Mira y elige, Haz la seña (cámara) |
| **Saludos** | Hola, Buenos días, Buenas tardes, Buenas noches, Gracias, Por favor, Bien, Adiós | Mira y elige, Elige la seña (video) |

- Lecciones cortas con XP, racha diaria y **vidas en forma de tazas de tinto** (3 por día; una respuesta mala derrama una taza).
- En los ejercicios con cámara, la app dibuja el esqueleto de la mano (21 puntos) y muestra una barra de precisión.
- Si fallas, **Nua** te da un consejo escrito para corregir la seña.
- Funciona en **modo claro y oscuro**.
- La app es para personas oyentes y sordas: ningún contenido depende del sonido.

---

## Estado del proyecto

**Modelos (Python, Google Colab):** terminados para letras, números, letras con movimiento y palabras.

**App (Flutter):**

| Fase | Qué | Estado |
|---|---|---|
| A | Configuración, fuentes, imágenes y modelos copiados | ✅ Hecha |
| B | Sistema de diseño: colores, tipografía, espaciado, tema claro/oscuro | ✅ Hecha |
| C | Componentes compartidos (botones, camino de lecciones, cámara, tarjetas…) | ✅ Hecha |
| D | Pantallas y reproductor de lecciones sin cámara | ⏳ Siguiente |
| E | Cámara nativa y detección de manos (Kotlin + MediaPipe) | Pendiente |
| F | Reconocimiento en Dart (puerto de los modelos) | Pendiente |
| G | Ejercicios con cámara y consejos de Nua | Pendiente |
| H | Progreso, perfil y créditos | Pendiente |
| I | Pulido, pruebas y APK final | Pendiente |
| J | (Opcional) Consejos de Nua con Gemini | Pendiente |

En el prototipo **no hay Firebase**: el progreso se guarda en el teléfono y los consejos vienen de un archivo de contenido.

---

## Estructura del repositorio

```
.
├─ ml/                         # Modelos de reconocimiento (Python)
│  ├─ fetch.py, remote_zip.py  # Descarga de datasets (en Colab, por rangos HTTP)
│  ├─ extract.py, qc.py        # Puntos de la mano con MediaPipe + control de calidad
│  ├─ normalize.py             # Contrato de características (fuente única de verdad)
│  ├─ sequences.py             # Secuencias para señas con movimiento
│  ├─ build_features.py        # Tablas de entrenamiento (.npz, se quedan en Drive)
│  ├─ train_letters.py         # Letras estáticas y números (MLP)
│  ├─ train_moving.py          # Letras con movimiento (MLP sobre secuencias)
│  ├─ train_words.py           # Palabras (MLP); dtw.py / words_dtw.py = versión DTW anterior
│  ├─ capture/                 # Captura de señas propias del equipo
│  ├─ demo_live/ demo_moving/ demo_numbers/ demo_numbers_team/ Demo_dynamic_words/
│  │                           # Demos con webcam (Windows, run.bat)
│  ├─ tests/                   # Pruebas del contrato de características
│  ├─ models/                  # Modelos de MediaPipe (.task, no se suben; ver README)
│  ├─ out/                     # Modelos exportados (JSON), golden files y reportes
│  ├─ soman_model.ipynb        # Cuaderno de Colab
│  └─ NOTES.md                 # Bitácora técnica de cada fase
└─ soman/                      # App Flutter (Android)
   ├─ lib/
   │  ├─ app/                  # Router y tema (colores, tipografía, espaciado)
   │  ├─ ui/                   # Componentes compartidos
   │  └─ features/             # Pantallas (debug/gallery por ahora)
   ├─ assets/
   │  ├─ fonts/                # Fraunces, Jost, Regular Aesthetic (logo)
   │  ├─ images/               # Nua (verde/amarilla), manos del inicio
   │  ├─ models/               # letters_mlp.json, moving_mlp.json, numbers_mlp.json
   │  └─ golden/               # Casos de prueba para el puerto a Dart
   └─ test/                    # Pruebas de contraste y de componentes
```

---

## Tecnologías y versiones

### App móvil

| Herramienta | Versión |
|---|---|
| Lenguaje | **Dart** 3.13.5 |
| Framework | **Flutter** 3.47.7 (canal estable) |
| Plataforma | Android, `minSdk 24` (requisito de MediaPipe Tasks) |
| Código nativo (fase E) | Kotlin + MediaPipe Tasks Vision (HandLandmarker) |
| Java (para compilar) | 17 |

| Paquete | Versión | Para qué |
|---|---|---|
| `flutter_riverpod` | 3.4.3 | Estado de la app |
| `go_router` | 18.1.1 | Navegación |
| `video_player` | 2.14.1 | Videos de referencia de cada seña |
| `shared_preferences` | 2.5.6 | Progreso guardado en el teléfono |
| `flutter_svg` | 2.3.0 | Íconos e ilustraciones |
| `confetti` | 0.8.0 | Celebración al terminar una lección |
| `flutter_lints` | 6.0.0 | Reglas de estilo (desarrollo) |

Las fuentes van incluidas en la app (no se descargan de internet).

### Modelos (Machine Learning)

| Herramienta | Versión | Para qué |
|---|---|---|
| **Python** | 3.12 (Colab y demos locales) | Todo el pipeline |
| **MediaPipe** HandLandmarker | `hand_landmarker.task` float16 v1 | 21 puntos 3D de cada mano |
| MediaPipe PoseLandmarker lite | `pose_landmarker_lite.task` float16 v1 | Hombros y nariz (solo palabras) |
| scikit-learn | 1.8 (Colab) / 1.9 (demos) | Redes MLP |
| NumPy, pandas, pyarrow | — | Datos y tablas |
| OpenCV (`opencv-python-headless`) | 5.0 | Lectura de video y webcam |
| Google Colab + Google Drive | — | Entrenamiento y almacenamiento de datos |

Dependencias en `ml/requirements.txt`.

---

## Cómo funciona el reconocimiento

1. **Un solo detector de manos en todas partes:** MediaPipe **HandLandmarker**, igual en los datasets y en el celular.
2. **Dos pasadas:** se detecta la mano en el cuadro completo, se recorta un cuadrado alrededor de ella (240×240) y se vuelve a detectar en el recorte. Los modelos de letras y números se entrenaron con recortes de mano, así que el celular hace lo mismo.
3. **Características:** `ml/normalize.py` convierte los 21 puntos en un vector normalizado (posición relativa a la muñeca, escala de la palma, mano izquierda reflejada). Es la **única fuente de verdad**: la versión en Dart debe dar los mismos números que los *golden files* de `ml/out/golden/` (tolerancia 1e-4).
4. **Clasificación:**
   - Letras estáticas y números: red MLP por cuadro; la seña cuenta cuando se sostiene 8 cuadros seguidos.
   - Letras con movimiento: MLP sobre 6 cuadros en 1 segundo (mismo ritmo que el dataset).
   - Palabras: MLP sobre la secuencia de la mano relativa a hombros y nariz.
5. **Siempre se evalúa con personas que el modelo nunca vio** (división por persona). Las grabaciones del equipo se usan como prueba final aparte.

Los modelos se exportan como **JSON** (pesos de la red) y se ejecutan en Dart puro: no hace falta TensorFlow Lite.

---

## Modelos y resultados

Todas las cifras son con **personas no vistas** en el entrenamiento.

| Modelo | Archivo | Clases | Resultado |
|---|---|---|---|
| Letras estáticas | `ml/out/letters_mlp.json` | 20 letras + "ninguna" | Exactitud media **0.93** (conjunto de 5 MLP) |
| Letras con movimiento | `ml/out/moving_mlp.json` | G J N Ñ Z (+ H S solo en video) | Acierto 0.80–0.96 en G J N Ñ Z; H y S no pasan |
| Números | `ml/out/numbers_mlp.json` | 1–5 + "ninguno" | Exactitud media **0.90**; grabaciones del equipo 0.86 |
| Palabras (saludos) | `ml/out/words_mlp_swap.json` | 8 saludos + "otra" | 6 de 8 palabras pasan (acierto medio 0.82); Gracias y Buenas noches no |

- Por ahora **Saludos se enseña solo con video** en la app. Los resultados con datasets distintos (otra cámara, otras personas) aún son bajos (~0.5–0.6).
- Los reportes completos, con matrices de confusión, están en `ml/out/*_report.md`, y el detalle de cada fase en `ml/NOTES.md`.

---

## Datasets

Los datos crudos **no están en este repositorio**: se descargan dentro de Colab y nunca se suben a Drive ni a GitHub. En Drive solo quedan tablas derivadas (puntos de la mano y características).

| Dataset | Uso | Contenido | Licencia | Fuente |
|---|---|---|---|---|
| **LSC70** | Letras y números | 70 personas, 37 clases (A–Z, Ñ, números), 6 cuadros por muestra; recortes de mano `LSC70ANH` | CC BY 4.0 | [Mendeley Data `9ssyn8tff5`](https://data.mendeley.com/datasets/9ssyn8tff5) |
| **LSC50** | Palabras y señas "impostoras" | 50 palabras, 5 personas × 4 repeticiones, video 1920×1080 | CC BY 4.0 | [Figshare `27383016`](https://doi.org/10.6084/m9.figshare.27383016) |
| **LSC-54** | Palabras | 43 palabras, 18 personas, 1.228 videos | **CC BY-NC 4.0** (no comercial) | [Science Data Bank](https://doi.org/10.57760/sciencedb.25639) |
| **Grabaciones del equipo** | Números 1–5 y prueba final | 3 personas del equipo, grabadas con `ml/capture/` | Propias | No publicadas |

**Palabras usadas** (IDs de LSC50): 0000 Gracias · 0001 Buenos días · 0002 Buenas tardes · 0003 Buenas noches · 0024 Bien · 0030 Hola · 0031 Adiós · 0032 Por favor. "Chao" de LSC-54 se une con Adiós.

> ⚠️ **LSC-54 es no comercial.** Cada plantilla de palabras lleva su campo `source`. Antes de cualquier uso comercial hay que quitar lo que venga de LSC-54 y reentrenar.

---

## Cómo ejecutarlo

### App (Android)

Requisitos: Flutter 3.47+, Android SDK (Android Studio), Java 17 y un celular Android 7.0+ con depuración USB activada.

```bash
cd soman
flutter pub get
flutter run            # con el celular conectado por USB
flutter test           # pruebas de contraste y de componentes
```

Por ahora la pantalla de inicio abre una **galería** (`/debug/gallery`) con todos los componentes en modo claro y oscuro.

> La fuente del logo (`assets/fonts/RegularAesthetic.ttf`) **no se sube al repositorio** porque su licencia aún no está confirmada. Pídela al equipo y ponla en `soman/assets/fonts/` antes de compilar.

### Demos con webcam (Windows)

```bash
cd ml
python -m venv .venv-demo
.venv-demo\Scripts\pip install -r requirements.txt opencv-python
```

Descarga los modelos de MediaPipe en `ml/models/` (enlaces en `ml/models/README.md`). Luego haz doble clic en el `run.bat` de cada demo:

- `demo_live/`: deletreo continuo (letras estáticas y con movimiento).
- `demo_moving/`: letras con movimiento.
- `demo_numbers_team/`: números 1–5.
- `Demo_dynamic_words/`: saludos.

### Entrenamiento (Google Colab)

1. Abre `ml/soman_model.ipynb` en Colab y monta Drive.
2. `python fetch.py lsc50 lsc54 lsc70 manifest` descarga los datasets en la máquina de Colab (~10 min).
3. `python extract.py …` y `python build_features.py` generan puntos y características (en Drive `soman-ml/`).
4. `python train_letters.py`, `train_letters.py --set numbers`, `train_moving.py` y `train_words.py` entrenan y exportan los modelos a `ml/out/`.

---

## Diseño

Estilo **"Frasurbane"**: cálido, editorial y redondeado, con referencias a pintura antigua.

- **Cinco colores:** crema `#FFF3B0`, vino `#540B0E`, rojo `#9E2A2B`, ocre `#E09F3E` y verde azulado `#335C67` (solo como acento).
- **Tipografía:** Regular Aesthetic (solo para el logo "SOMAN"), Fraunces itálica (títulos) y Jost (texto).
- Botones en forma de píldora, barra de navegación flotante, sin bordes, sin degradados y sin emojis.
- Todo el texto cumple un contraste mínimo de 4.5:1, verificado con pruebas automáticas.

---

## Licencias y créditos

- **Código:** licencia por definir.
- **Datos:** LSC70 y LSC50 bajo CC BY 4.0; LSC-54 bajo CC BY-NC 4.0. Gracias a sus autores.
- **Fuentes:** Fraunces y Jost bajo SIL Open Font License (archivos `*-OFL.txt`). Regular Aesthetic ("Aesthetic", © 2022 Nirmana Visual): licencia por confirmar.
- **MediaPipe:** © Google, Apache 2.0.
- **Ilustraciones y mascota (Nua):** equipo Soman.

Hecho con ☕ por el equipo Soman para el Hackathon GDG.
