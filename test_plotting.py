import utils

VIDEO = 'holsten_gate'
OBSERVER = 'AAW' # AAF

video_path = f'data/GazeCom/movies_m2t/{VIDEO}.m2t'
label_path = f'data/GazeCom/deepEM_classifier/ground_truth_framewise/{VIDEO}/{OBSERVER}_{VIDEO}.txt'
dir_arff_ground_truth = r'data/GazeCom/deepEM_classifier/ground_truth'
raw_label_path = f'{dir_arff_ground_truth}/{VIDEO}/{OBSERVER}_{VIDEO}.arff'

out_path = f'data/GazeCom/movies_with_groundtruth'
out_path_frames = f'{out_path}/{OBSERVER}_{VIDEO}'

utils.plot_gazecom_frames_with_labels(video_path, label_path, raw_label_path, save_to_directory=out_path_frames)
if out_path is not None:
    utils.create_movie_from_frames(out_path, f'{OBSERVER}_{VIDEO}', f'{OBSERVER}_{VIDEO}.mp4', fps=10, width_px=1280)
