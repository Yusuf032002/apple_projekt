\# 🍎 Detektion af rådne æbler – Computer Vision



Dette projekt anvender klassiske computer vision-teknikker til at identificere og adskille

\*\*friske\*\* og \*\*rådne æbler\*\* ud fra billeder.



Projektet er lavet som en del af computer vision-undervisningen og følger pensum.



---



\## 🎯 Formål

\- Automatisk detektion af rådne æbler

\- Reducere manuel sortering

\- Demonstrere brug af klassiske CV-metoder



---



\## 🧠 Anvendte teknikker (pensum)

Projektet anvender følgende computer vision-metoder:



\- \*\*Farverum\*\*

&nbsp; - BGR → HSV (brug af S-kanalen til segmentering)

\- \*\*Thresholding\*\*

&nbsp; - Global threshold

&nbsp; - Adaptive threshold (mean \& gaussian)

&nbsp; - Otsu

\- \*\*Edge detection\*\*

&nbsp; - Sobel

&nbsp; - Canny

&nbsp; - Laplace

\- \*\*Konturdetektion\*\*

&nbsp; - `cv2.findContours`

&nbsp; - Filtrering baseret på areal og aspektforhold

\- \*\*Blob detection\*\*

&nbsp; - `cv2.SimpleBlobDetector`

&nbsp; - Detektion af defekte pletter

\- \*\*Morfologiske operationer\*\*

&nbsp; - Opening / Closing

&nbsp; - Erosion (inner mask)



---



\## ▶️ Sådan køres projektet



\### Standard (ingen pop-ups)

```bash

python detect\_rotten\_apples.py



Vis billeder på skærmen

python detect\_rotten\_apples.py --show



Gem analyse-/debug-billeder

python detect\_rotten\_apples.py --analysis



apple\_projekt/

├── detect\_rotten\_apples.py

├── images/

│   ├── apple1.jpg

│   ├── apple2.jpg

│   └── ...

├── output/

├── debug/

└── README.md

🧪 Output



Klassifikation af hvert æble som FRISK eller RÅDDENT



Resultatbilleder gemmes i output/



Analysebilleder (threshold, edges, blobs) gemmes i debug/



🛠 Krav



Python 3.x



OpenCV (opencv-python)



NumPy



Installer afhængigheder:

pip install opencv-python numpy





