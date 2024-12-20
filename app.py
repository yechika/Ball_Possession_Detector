from flask import Flask, request, jsonify, send_file, render_template
from flask_socketio import SocketIO
import os
from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

app = Flask(__name__)
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploaded_videos'
OUTPUT_FOLDER = 'output_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    video_file = request.files['video']
    input_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, 'output_video.avi')

    # Save the uploaded video
    video_file.save(input_path)

    # Process the video in a background thread
    socketio.start_background_task(process_video, input_path, output_path)

    return jsonify({'message': 'Processing started'})

@app.route('/download/<filename>')
def download_video(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

def process_video(input_path, output_path):
    def update_progress(message, percentage):
        socketio.emit('progress', {'message': message, 'percentage': percentage})

    update_progress("Starting the video processing pipeline...", 0)

    # Step 1: Read Video
    video_frames = read_video(input_path)
    tracker = Tracker('models/best.pt')

    # Step 2: Tracking objects
    update_progress("Tracking objects...", 10)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Step 3: Add positions to tracks
    update_progress("Adding positions to tracks...", 20)
    tracker.add_position_to_tracks(tracks)

    # Step 4: Estimate camera movement
    update_progress("Estimating camera movement...", 30)
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')

    update_progress("Adjusting positions based on camera movement...", 40)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 5: Transform view
    update_progress("Transforming view...", 50)
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Step 6: Interpolate ball positions
    update_progress("Interpolating ball positions...", 60)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 7: Estimate speed and distance
    update_progress("Estimating speed and distance...", 70)
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 8: Assign player teams
    update_progress("Assigning player teams...", 80)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Step 9: Assign ball acquisition
    update_progress("Assigning ball acquisition...", 90)
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team = tracks['players'][frame_num][assigned_player].get('team', 'unknown')  # Default: 'unknown'
            team_ball_control.append(team)
        else:
            # Use the last known team control or default to 'none'
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 'none')

    team_ball_control = np.array(team_ball_control)

    # Step 10: Draw annotations
    update_progress("Drawing annotations...", 95)
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Step 11: Save video
    update_progress("Saving video...", 100)
    save_video(output_video_frames, output_path)

    update_progress("Processing completed!", 100)

if __name__ == '__main__':
    socketio.run(app, debug=True)
