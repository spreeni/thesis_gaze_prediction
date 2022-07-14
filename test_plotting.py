"""
Script to visualize ground truth labels over video.
"""
import utils

VIDEO = 'holsten_gate'
OBSERVER = 'AAW'

video_path = f'data/GazeCom/movies_m2t/{VIDEO}.m2t'
label_dir = f'data/GazeCom/deepEM_classifier/ground_truth_framewise/{VIDEO}'
label_path = f'{label_dir}/{OBSERVER}_{VIDEO}.txt'
dir_arff_ground_truth = r'data/GazeCom/deepEM_classifier/ground_truth'
raw_label_path = f'{dir_arff_ground_truth}/{VIDEO}/{OBSERVER}_{VIDEO}.arff'

out_path = f'data/GazeCom/movies_with_groundtruth'
out_path_frames_dir = f'{OBSERVER}_{VIDEO}_all'

utils.plot_gazecom_frames_with_all_observers(video_path, label_dir, plot_em_data=False, save_to_directory=f'{out_path}/{out_path_frames_dir}',
                                             n_observers=None)
#utils.plot_gazecom_frames_with_labels(video_path, label_path, raw_label_path, save_to_directory=out_path_frames)
if out_path is not None:
    movie_name = f'{OBSERVER}_{VIDEO}_all_observers'
    utils.create_movie_from_frames(out_path, out_path_frames_dir, f'{movie_name}.mp4', fps=10, width_px=1280, delete_frames=False)
    utils.create_movie_from_frames(out_path, out_path_frames_dir, f'{movie_name}_30fps.mp4', fps=30, width_px=1280, delete_frames=False)
