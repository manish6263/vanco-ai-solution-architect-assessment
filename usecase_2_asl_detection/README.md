# Use Case 2: American Sign Language Detection

## Objective

Collect a custom American Sign Language image dataset, annotate hand bounding boxes, train an object detection model, and run live webcam inference.

## Planned Architecture

```text
Webcam image collection
    -> annotation in YOLO format
    -> train/validation/test split
    -> YOLO object detection training
    -> metric reporting
    -> live webcam inference app
```

## Scope

Minimum assessment requirement:

- At least 8 ASL classes
- Minimum 20 images per class
- Bounding boxes around the hand region
- Live webcam demo with bounding box, predicted sign, confidence score, and acceptable latency

Preferred target:

- 10-12 ASL classes
- 40-60 images per class if time permits
- Multiple backgrounds and lighting conditions
- At least one signer-independent or background-independent validation split if possible

## Dataset Plan

Dataset folders will be organized as:

```text
dataset/
  raw/
  processed/
  data.yaml
annotation_samples/
```

The dataset summary should include:

- Class list
- Image counts per class
- Split counts
- Annotation format
- Collection conditions
- Known limitations

## Model Plan

Primary model: Ultralytics YOLO nano/small variant.

Reasons:

- Fast enough for webcam inference
- Strong object detection baseline
- Simple deployment path
- Clear metrics and visual outputs

## Evaluation Plan

Report:

- mAP
- Precision
- Recall
- Confusion matrix
- Per-class performance
- Webcam FPS/latency
- Failure cases

## Demo Plan

The live demo should:

- Open webcam input
- Draw bounding boxes
- Display predicted ASL class
- Display confidence score
- Display FPS or latency
- Smooth predictions across recent frames if needed

## Deliverables

- [ ] Custom dataset summary
- [ ] Annotation samples
- [ ] Trained model weights
- [ ] Evaluation metrics
- [ ] Webcam demo script
- [ ] Backup recorded demo
- [ ] Architecture diagram
- [ ] Deployment and robustness notes

