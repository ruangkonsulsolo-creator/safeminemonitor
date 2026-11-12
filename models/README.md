Put the trained PPE model here as **best.pt** (see training/train_yolo.py or training/train_colab.ipynb).
Without best.pt the app falls back to yolo11n.pt (COCO): only RESTRICTED_ZONE and MACHINE_PROXIMITY work.
You can also point to another file: set environment variable SAFEMINE_MODEL=path/to/model.pt
