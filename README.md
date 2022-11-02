# Binary Image Classifier

This repository focuses on a binary image classifier that aims to classify military and civilian vehicles, using
a large MobileNet-V3 architecture.

## File Structure
```
+-- app
|   +-- __init__.py
|   +-- cleaning.py
|   +-- imaging.py
|   +-- model.py
|   +-- plotter.py
|   +-- utils.py
+-- labels
|   +-- test_labels.csv
|   +-- train_labels.csv
|   +-- updated_labels.csv
+-- saved_models
|   +-- mobilenetv3.pt
+-- tests
|   +-- __init__.py
|   +-- test_cleaning.py
+-- demo.ipynb
+-- parameters.toml
+-- README.md
+-- requirements.txt
```

- `/app` - houses the functionality of the classifier
  - `cleaning.py` - covers functionality for cleaning the label data
  - `imaging.py` - includes functionality for moving and changing image file names
  - `model.py` - involves all model related functionality
  - `plotter.py` - plotting functionality
  - `utils.py` - utility classes and functions used throughout the app

