import pickle
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import os
from utils import get_center_of_bbox, get_bbox_width
import pandas as pd


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        # Edge case if the missing detection is first one it won't interpolate it so we can backfill it which nearest detection we can find
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        # Process frames in batches to optimize performance
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(
                batch_frames, conf=0.2, save=False)
            detections.extend(batch_detections)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path and os.path.exists(stub_path):
            print("Reading tracks from stub...")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {
            # {0: {"bbox": [0,0,0,0]}, 1: {"bbox": [0,0,0,0]}, 21: {"bbox": [0,0,0,0]}},    # Frame 1
            # {10: {"bbox": [0,0,0,0]}, 21: {"bbox": [0,0,0,0]}}   # Frame 2
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # Invert the class names dictionary from person: 0 to 0: person
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert the supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

             # Filter out ball detection for tracking as we don't want to track the ball since there is only one instance of ball
            non_ball_mask = detection_supervision.class_id != cls_names_inv['ball']
            detection_for_tracking = detection_supervision[non_ball_mask]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_for_tracking)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox, "conf": frame_detection[2].tolist()}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {
                        "bbox": bbox, "conf": frame_detection[2].tolist()}

            # Don't want to track the ball so looping over the original detections to get the ball bbox
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                print(frame_detection)

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox, "conf": frame_detection[2].tolist()}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None, conf=None):
        y2 = int(bbox[3])
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            img=frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + int(0.35*width)
        y2_rect = (y2 + rectangle_height // 2) + int(0.35*width)

        if track_id:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2)

        if conf:
            conf_text = f"{conf:.2f}"
            x1_text = x1_rect + 12

            cv2.putText(
                frame,
                conf_text,
                (int(x1_text), int(y2_rect + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color, conf=None):
        y = int(bbox[1])
        x_center, y_center = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x_center, y],
            [x_center - 10, y - 20],
            [x_center + 10, y - 20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        if conf:
            conf_text = f"{conf:.2f}"
            cv2.putText(
                frame,
                conf_text,
                (x_center, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, player["bbox"],  colour, track_id, conf=player.get("conf", None))

            # Draw Referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, referee["bbox"],  (0, 255, 255), track_id=None, conf=referee.get("conf", None))

            # Draw Ball
            for ball_id, ball in ball_dict.items():
                print("Ball: ", ball)

                frame = self.draw_triangle(
                    frame, ball["bbox"], (0, 255, 0), conf=ball.get("conf", None))

            output_video_frames.append(frame)

        return output_video_frames
