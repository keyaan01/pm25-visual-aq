# Glossary (grows as the project does)

Plain-language meanings for the terms you'll meet. For deeper explanations, the
`PM2.5_Project_Guide_Explained.pdf` you already have covers most of these in detail.

| Term | Plain meaning |
|------|---------------|
| **PM2.5** | Airborne particles ≤ 2.5 µm — small enough to reach deep into the lungs. |
| **AQI (Air Quality Index)** | A unitless 0–500+ index that remaps pollutant concentrations onto public-facing bands. **Our label is AQI, not µg/m³.** |
| **µg/m³** | Micrograms per cubic metre — the raw concentration unit (used only in the C2 hourly data). |
| **Label** | The answer the model is trained to predict — here, the AQI of a photo. |
| **Feature** | An input the model sees — here, the 5 image channels. |
| **Data leakage** | When info from the test set sneaks into training, making scores look better than reality. |
| **Near-duplicate** | Two photos that look almost identical (e.g. consecutive street frames). |
| **Perceptual hash** | A fingerprint where *similar images get similar fingerprints*, used to find near-duplicates. |
| **Station-grouped split** | Putting all photos from one monitoring station entirely in train *or* test, never both — prevents leakage. |
| **Transmission** | The fraction of a scene's light that survives the trip to the camera; low when haze is dense. |
| **Dark Channel Prior (DCP)** | A haze-detection trick: clear photos have a very dark pixel in every small patch; haze lifts it. |
| **Inverted saturation** | A map that is high where colour is washed out — works on sky, where DCP fails. |
| **CNN / backbone** | A neural network built for images; the shared "trunk" that turns a photo into a feature summary. |
| **EfficientNet-B0** | The specific, compact (5.3 M-parameter) pretrained CNN we use as the backbone. |
| **Quantile / percentile** | A cutoff in a distribution; the 5th/50th/95th percentiles form our low/median/high range. |
| **Pinball loss** | The asymmetric loss that trains a head to predict a specific percentile. |
| **Monotone quantile heads** | Building the 3 outputs so low ≤ median ≤ high always holds, by construction. |
| **Log-transform** | Training on `log(AQI)` because the label is right-skewed; we invert it at prediction time. |
| **Conformal prediction / CQR** | A calibration step that makes the interval contain the truth a guaranteed % of the time. |
| **Coverage** | The % of test cases where the true value actually landed inside the predicted interval. |
| **Calibration set** | A held-out slice used only to size the conformal correction — never for training. |
| **Abstention / reject option** | Letting the model refuse to answer on inputs it can't handle (night, indoor, no sky). |
| **Risk–coverage curve** | A plot of accuracy vs. how often the model chooses to answer. |
| **Error ceiling / Bayes error** | The best score any model could get, limited by noise in the labels themselves. |
| **MAE / RMSE / R²** | Standard regression scores (average error / error that punishes big misses / fraction of variation explained). |
| **Ablation** | Removing one component at a time to measure how much it actually helps. |
