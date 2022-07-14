"""
Sample predictions over many trained models at once.

Visualize results and calculate metrics like NSS or gaze change similarities.
"""
from tqdm.auto import tqdm

from sample_model import sample_model


_SHOW_SALIENCY = True
_PLOT_GAZE_CHANGE_HISTOGRAMS = True

_TEACHER_FORCING_ON_INFERENCE = True

pbar_outer = tqdm([
    # dataset in ['single_clip', 'single_video', 'single_video_all_observers', 'all_videos_single_observer', 'all_videos_all_observers']
    (r'data/lightning_logs/version_534/checkpoints/epoch=149-step=149.ckpt', r'data/sample_outputs/version_534_teacher', 'all_videos_single_observer'), #all_videos_all_observers
    #(r'data/lightning_logs/version_533/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_533', 'single_clip'), #all_videos_all_observers
    #(r'data/lightning_logs/version_536/checkpoints/epoch=135-step=543.ckpt', r'data/sample_outputs/version_536', 'single_video_all_observers'), #all_videos_all_observers
    #(r'data/lightning_logs/version_549/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_549', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_550/checkpoints/epoch=188-step=188.ckpt', r'data/sample_outputs/version_550', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_552/checkpoints/epoch=193-step=193.ckpt', r'data/sample_outputs/version_552', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_551/checkpoints/epoch=140-step=140.ckpt', r'data/sample_outputs/version_551', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_548/checkpoints/epoch=101-step=403.ckpt', r'data/sample_outputs/version_548', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_546/checkpoints/epoch=170-step=341.ckpt', r'data/sample_outputs/version_546', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_530/checkpoints/epoch=148-step=148.ckpt', r'data/sample_outputs/version_530_500', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_556/checkpoints/epoch=32-step=1655.ckpt', r'data/sample_outputs/version_556', 'all_videos_all_observers'),
    #(r'data/lightning_logs/version_557/checkpoints/epoch=226-step=226.ckpt', r'data/sample_outputs/version_557', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_558/checkpoints/epoch=229-step=229.ckpt', r'data/sample_outputs/version_558', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_559/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_559', 'single_video'), #all_videos_all_observers
    #(r'data/lightning_logs/version_564/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_564', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_565/checkpoints/epoch=156-step=156.ckpt', r'data/sample_outputs/version_565', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_562/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_562', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_563/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_563', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_566/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_566', 'all_videos_single_observer'),
    #(r'data/lightning_logs/version_572/checkpoints/epoch=200-step=200.ckpt', r'data/sample_outputs/version_572', 'all_videos_single_observer', 2),
    #(r'data/lightning_logs/version_573/checkpoints/epoch=200-step=401.ckpt', r'data/sample_outputs/version_573', 'all_videos_single_observer', 10),
    #(r'data/lightning_logs/version_574/checkpoints/epoch=161-step=161.ckpt', r'data/sample_outputs/version_574', 'single_clip', 1.5),
    #(r'data/lightning_logs/version_579/checkpoints/epoch=191-step=191.ckpt', r'data/sample_outputs/version_579', 'all_videos_single_observer'),
    #(r'log', r'data/sample_outputs/version_5xx', 'all_videos_single_observer')
])
for _CHECKPOINT_PATH, _OUTPUT_DIR, partition in pbar_outer:
    _CLIP_DURATION = 5 if partition != 'single_clip' else 2
    
    for _MODE, _CALC_METRICS in [
        #('train', False),
        #('val', False),
        ('train', True),
        ('val', True),
    ]:
        pbar_outer.set_description(f"{_OUTPUT_DIR.split('/')[-1]}, _MODE: {_MODE}, _CALC_METRICS: {_CALC_METRICS}")
        sample_model(_CHECKPOINT_PATH, _OUTPUT_DIR, partition, _CLIP_DURATION, _MODE, _CALC_METRICS,
                     _SHOW_SALIENCY, _PLOT_GAZE_CHANGE_HISTOGRAMS, _TEACHER_FORCING_ON_INFERENCE)
