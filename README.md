# Testing Suite for Algorithmic, CV, and Image recognition model-based acne detection systems
Open source project developed in collaboration with ML@Purdue and Skinapse Labs

Each detector is a separate method. Generated images, trained models, and calibration dumps stay on disk and are not committed.

| Method | Script | Local outputs |
|---|---|---|
| Original heatmap | `original_heatmap_detector.py` | `output/original_heatmap/` |
| Simple heatmap | `simple_heatmap_detector.py` | `output/simple_heatmap/`, `models/simple_heatmap/` |
| Basic heatmap | `basic_heatmap_detector.py` | `output/basic_heatmap/`, `calibration/basic_heatmap/` |
| Data-driven heatmap | `datadriven_heatmap_detector.py` | `output/datadriven_heatmap/`, `calibration/datadriven_heatmap/` |
| Individual spots | `individual_spot_detector.py` | `output/individual_spots/`, `calibration/individual_spots/` |
| Sklearn ML | `sklearn_ml_detector.py` | `output/sklearn_ml/`, `models/sklearn_ml/` |
| Pixel analysis | `pixel_analysis_detector.py` | `output/pixel_analysis/single/`, `output/pixel_analysis/dataset_inference_v2/`, `calibration/pixel_analysis/` |

Earlier unlabeled ML runs are kept in `output/early_ml/` and `models/early/`. The first pixel-analysis dataset pass is in `output/pixel_analysis/dataset_inference_v1/`.

Put a test photo at `input/testImage.png`. Install dependencies with `pip install -r requirements.txt`.
