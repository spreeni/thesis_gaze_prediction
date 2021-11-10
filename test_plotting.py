import utils

VIDEO = 'breite_strasse'
OBSERVER = 'ALK'

video_path = f'data/GazeCom/movies_mpg/{VIDEO}.mpg'
label_path = f'data/GazeCom/movies_mpg_frames/train/label_data/{VIDEO}/{OBSERVER}_{VIDEO}.txt'
dir_arff_ground_truth = r'data/GazeCom/deepEM_classifier/ground_truth'
raw_label_path = f'{dir_arff_ground_truth}/{VIDEO}/{OBSERVER}_{VIDEO}.arff'

utils.plot_gazecom_frames_with_labels(video_path, label_path, raw_label_path)
