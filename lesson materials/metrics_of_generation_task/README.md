# Лекция №6. Метрики оценки моделей
В лекции рассматривается ряд метрик для оценки качества генеративных моделей и изображений.

### Классификация метрик
Метрики оценки генеративных моделей:
+ Основанных на правдоподобии (likelihood-based):
    - Правдоподобие
+ Неявной плотности (likelihood-free):
    - IS (Inception score)
    - FID (Frechet Inception Distance)
    - Precision-Recall

Метрики оценки качества изображений:
- Low-level
    - PixCorr (Pixelwise Correlation)
    - SSIM (Structural Similarity Index Measure)
- High-level
    - CLIP (Contrastive Language-Image Pre-Training)
    - SwAV (Swapping Assignments between multiple Views)

- Метрики, не рассмотренные в лекции:
    - Mean Squared Error (MSE) & Root Mean Squared Error (RMSE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
    - Spatial Correlation Coefficient (SCC)
    - Relative Average Spectral Error (RASE)
    - Spectral Angle Mapper (SAM)
    - Visual Information Fidelity (VIF)


### Директории/Файлы:
- `metrics_of_generation_task.ipynb` - блокнот с текстом и кодом лекции
- `./imgs/original` - образцы реальных изображений
- `./imgs/generated` - образцы сгенерированных изображений

### Литература
- [Исаченко Р. Порождающие модели машинного обучения. МФТИ, 2023](https://www.youtube.com/playlist?list=PLk4h7dmY2eYHVCEMMMqdKes__ehs5mRtR)
- [Stefano Ermon. Deep Generative Models. Stanford](https://deepgenerativemodels.github.io/)
- [David Mack. A simple explanation of the Inception Score. Medium](https://medium.com/octavian-ai/a-simple-explanation-of-the-inception-score-372dff6a8c7a)
- [Salimans T. Improved Techniques for Training GANs, 2016](https://arxiv.org/abs/1606.03498)
- [Heusel M. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. 2017](https://arxiv.org/abs/1706.08500)
- [Kynkäänniemi T. Improved Precision and Recall Metric for Assessing Generative Models. 2019](https://arxiv.org/abs/1904.06991)
- [Zhou Wang. Image Quality Assessment: From Error Visibility to Structural Similarity. 2004](https://www.researchgate.net/publication/3327793_Image_Quality_Assessment_From_Error_Visibility_to_Structural_Similarity)
- [Radford A. Learning Transferable Visual Models From Natural Language Supervision. 2021](https://arxiv.org/pdf/2103.00020) \
[CLIP. GitHub](https://github.com/OpenAI/CLIP)

Дополнительно:
- [Universal Quality Image Index (UQI)](https://ieeexplore.ieee.org/document/995823)
- [Complex Wavelet SSIM (CW-SSIM)](https://ieeexplore.ieee.org/document/5109651)
- [Multi-scale Structural Similarity Index (MS-SSIM)](https://ieeexplore.ieee.org/abstract/document/1292216)