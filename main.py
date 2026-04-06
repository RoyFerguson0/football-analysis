import cv2
import numpy as np

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator


def main():
    # Read video
    video_frame = read_video('input_videos/football_game1.mp4')
    print(f"Read {len(video_frame)} frames from the video.")

    # Initialize tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(
        video_frame, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    # region Save cropped image of player
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player["bbox"]
    #     frame = video_frame[0]

    #     # Crop bbox from frame
    #     cropped_image = frame[
    #         int(bbox[1]):int(bbox[3]),
    #         int(bbox[0]):int(bbox[2])]

    #     cv2.imwrite(f"output_videos/cropped_img.jpg", cropped_image)
    #     break
    # endregion

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frame[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frame, read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl")

    # Adjust player/ball positions based on camera movement
    camera_movement_estimator.add_adjuct_positions_to_tracks(
        tracks, camera_movement_per_frame)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize team assigner
    team_assigner = TeamAssigner()

    # Assign team colors to players
    team_assigner.assign_team_color(video_frame[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frame[frame_num],
                                                 track["bbox"],
                                                 player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_colour"] = team_assigner.team_colours[team]

    # Assign Ball Aquisition to Player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_players(
            player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(
                team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output and Draw object tracks
    output_video_frames = tracker.draw_annotations(
        video_frames=video_frame, tracks=tracks, team_ball_control=team_ball_control)

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.mp4')
    print("Video saved successfully.")


if __name__ == "__main__":
    main()
