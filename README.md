# 3D Facial Expression Recognition

This project performs facial expression recognition using 3D facial landmark data. The goal is to classify facial expressions using machine learning and compare performance across original, translated, and rotated landmark data.

## Project Overview

This project uses 3D facial landmarks from the BU4DFE dataset. Each face sample contains landmark points with x, y, and z coordinates. The model uses these landmark features to classify facial expressions such as Angry, Disgust, Fear, Happy, Sad, and Surprise.

The project applies Leave-One-Subject-Out (LOSO) cross-validation to evaluate the model. Random Forest classifier is used for classification.

## Features

- Reads 3D facial landmark `.bnd` files
- Extracts x, y, and z landmark coordinates
- Supports original landmark data
- Supports translated landmark data
- Supports rotated landmark data across X, Y, and Z axes
- Uses LOSO cross-validation
- Prints accuracy, precision, recall, and confusion matrix
- Saves experiment results

## Classifier Used

Random Forest classifier is used in this project. Random Forest builds multiple decision trees and combines their predictions using majority voting. It works well with high-dimensional data such as 3D facial landmarks and helps to reduce overfitting.

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to run
```bash
python Project1.py <data_type> <data_directory>
```

## Data Type Options

| Option | Description |
|---|---|
| `o` | Original data |
| `t` | Translated data |
| `x` | Rotated over X-axis |
| `y` | Rotated over Y-axis |
| `z` | Rotated over Z-axis |

## Example Commands

### Translated Data

```bash
python Project1.py t ./dataset/BU4DFE_BND_V1.1
```
## Results Summary

The translated data performed best in this project. It achieved the highest accuracy, precision, and recall compared to original and rotated data.

| Data Type | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Original | 0.3618 | 0.3577 | 0.3626 |
| Translated | 0.4838 | 0.4749 | 0.4843 |
| Rotated X | 0.3368 | 0.3269 | 0.3375 |
| Rotated Y | 0.3412 | 0.3367 | 0.3420 |
| Rotated Z | 0.3409 | 0.3326 | 0.3416 |

## Conclusion

The translated landmark data gave the best performance because it centers the face around the origin and reduces position-based differences. This helps the classifier focus more on facial expression patterns instead of face location.

## Author

Pearl Viralkumar Patel






