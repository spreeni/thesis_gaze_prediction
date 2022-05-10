import utils

VIDEO = 'holsten_gate'
OBSERVER = 'AAW' # AAF

video_path = f'data/GazeCom/movies_m2t/{VIDEO}.m2t'
label_path = f'data/GazeCom/deepEM_classifier/ground_truth_framewise/{VIDEO}/{OBSERVER}_{VIDEO}.txt'
dir_arff_ground_truth = r'data/GazeCom/deepEM_classifier/ground_truth'
raw_label_path = f'{dir_arff_ground_truth}/{VIDEO}/{OBSERVER}_{VIDEO}.arff'

out_path = f'data/GazeCom/movies_with_groundtruth/{OBSERVER}_{VIDEO}'

utils.plot_gazecom_frames_with_labels(video_path, label_path, raw_label_path, save_to_directory=out_path)
