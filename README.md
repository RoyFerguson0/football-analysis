# Football Analysis

Computer vision pipeline for football match analysis. The project detects and tracks players, referees, and the ball from broadcast footage, estimates camera motion with optical flow, separates teams using KMeans clustering on jersey colors, maps image coordinates to a top-down pitch view with perspective transformation, and computes player speed and distance in real-world units.

This repository is designed to show an end-to-end applied Computer Vision workflow rather than a single model demo. It combines object detection, multi-object tracking, motion compensation, clustering, homography, and simple possession logic into one video-processing pipeline.

## Demo Video

**[Watch the demo video](assets/quick-demo-vid.mp4)**

[![Watch the demo](assets/thumbnail.png)](assets/quick-demo-vid.mp4)

## What the project does

- Detects players, referees, and the ball with a custom YOLO model.
- Tracks detections across frames with ByteTrack.
- Estimates camera pan and shake using Lucas-Kanade optical flow.
- Compensates player positions for camera movement so movement is measured more accurately.
- Uses KMeans clustering to infer team colors from player jerseys.
- Applies perspective transformation to convert image coordinates into pitch coordinates.
- Estimates player speed and distance travelled on the field.
- Assigns ball possession to the nearest player and visualizes team ball control over time.

## Technical Highlights

### Object detection and tracking

The pipeline starts with a YOLO detector trained on football-specific classes. Detections are passed through ByteTrack to maintain stable player and referee IDs across frames. The tracker also handles ball detections separately so the ball can be interpolated when detections are missing.

### Optical flow for camera movement

Football broadcast footage often includes pans and zooms, which can make stationary players appear to move. To address this, the project estimates frame-to-frame camera motion with Lucas-Kanade optical flow. Good feature points are detected in the first frame, tracked forward through time, and aggregated into a median camera shift for each frame.

That motion estimate is then subtracted from player positions so downstream measurements are less sensitive to camera movement.

### Team classification with KMeans

Rather than relying on the detector to identify team affiliation, the project samples jersey colors from detected player crops and uses KMeans clustering to group players into two teams. This is a useful example of combining unsupervised learning with vision features extracted from detections.

### Perspective transformation

The pitch is mapped from image space into a top-down court representation using a perspective transform. Once player positions are projected into this plane, the project can estimate movement in meters instead of pixels.

### Speed and distance estimation

Using the transformed positions and frame rate, the project computes:

- instantaneous player speed in km/h
- cumulative distance travelled in meters

### Ball possession logic

Ball possession is assigned using a proximity-based heuristic: the ball is matched to the closest player foot position when the distance is below a threshold. The result is also summarized as team ball control over time.

## Pipeline Overview

1. Load the input video into frames.
2. Run YOLO detection on each frame.
3. Track players, referees, and the ball with ByteTrack.
4. Compute player foot positions and ball center positions.
5. Estimate camera movement with optical flow.
6. Adjust tracked positions using the camera motion estimate.
7. Project positions onto a top-down pitch with perspective transformation.
8. Estimate player speed and distance.
9. Assign team labels with KMeans clustering on jersey colors.
10. Estimate ball possession and render visual annotations.
11. Save the final annotated video.

## Repository Structure

- `main.py` - Orchestrates the full analysis pipeline.
- `trackers/` - YOLO detection, tracking, interpolation, and visualization utilities.
- `camera_movement_estimator/` - Optical flow based camera motion estimation.
- `team_assigner/` - Jersey-color clustering and team assignment.
- `player_ball_assigner/` - Ball possession assignment.
- `view_transformer/` - Perspective transform from image space to pitch space.
- `speed_and_distance_estimator/` - Speed and distance calculations.
- `utils/` - Video and geometry helper functions.
- `training/` - Model training notebook, dataset exports, and training artifacts.

## Installation

This project uses `uv` and Python packages listed in `pyproject.toml`.

```bash
uv sync
```

The project expects a CUDA-capable PyTorch setup when GPU acceleration is available. The locked environment uses the PyTorch cu128 wheels.

## Running the Analysis

Place a football video in `input_videos/` and update the path in `main.py` if needed.

```bash
uv run main.py
```

The annotated output is written to `output_videos/output_video.mp4`.

## Output Visualizations

The generated video includes:

- player and referee bounding boxes
- ball annotations
- team colors and team IDs
- ball possession overlays
- camera movement text
- player speed and distance labels

## Notes on the Implementation

- Camera movement is estimated from strong image features using optical flow, then filtered with a minimum movement threshold.
- Team labels are assigned from jersey colors, not from the detector class labels.
- Ball position is interpolated when detections are missing so short occlusions do not break the pipeline.
- Pitch coordinates are computed from a fixed set of manually chosen source and target points.

## Why this project is useful

This is a compact example of a real CV system that goes beyond inference on a single frame. It demonstrates detection, tracking, optical flow, clustering, geometry, and temporal reasoning in one pipeline, which is exactly the kind of work that is relevant for applied machine learning and computer vision roles.
