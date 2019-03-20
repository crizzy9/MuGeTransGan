import numpy as np
import torch


def to_numpy(multitrack_pianoroll, expected_tracks):
    num_tracks = len(multitrack_pianoroll.tracks)
    if expected_tracks is not None and num_tracks != expected_tracks:
        raise RuntimeError(f'Given pianoroll with {num_tracks} tracks, expected {self.expected_tracks}')
    return np.array([track.pianoroll for track in multitrack_pianoroll.tracks])


def to_padded(tracks, pad_to, pad_token):
    num_tracks, input_len, input_width = tracks.shape
    if input_len == pad_to:
        return tracks
    if input_len > pad_to:
        raise RuntimeError(f'Given pad_to {self.pad_to}, but input has length {input_len}')
    padding = np.ones((num_tracks, pad_to - input_len, input_width)) * pad_token

    return np.hstack((tracks, padding))

def total_transform(pad_to=959040, pad_token=129, expected_tracks=5):
    return lambda x : torch.tensor(to_padded(to_numpy(x, expected_tracks), pad_to, pad_token))

    

    

        
    
    
        
        
            
            