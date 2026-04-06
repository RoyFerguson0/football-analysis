from utils import read_video, save_video
from trackers import Tracker


def main():
    # Read video
    video_frame = read_video('input_videos/football_game1.mp4')
    print(f"Read {len(video_frame)} frames from the video.")

    # Initialize tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(
        video_frame, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    # Draw output and Draw object tracks
    output_video_frames = tracker.draw_annotations(
        video_frames=video_frame, tracks=tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.mp4')
    print("Video saved successfully.")


if __name__ == "__main__":
    main()
