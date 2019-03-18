import numpy as np

class ToPadded(object):
    
    def __init__(self, pad_to, pad_token):
        self.pad_to = pad_to
        self.pad_token = pad_token
        
    def __call__(self, input_pianoroll):
        multitrack_pianoroll = input_pianoroll.copy()
        input_len, input_width = multitrack_pianoroll.tracks[0].pianoroll.shape
        
        if input_len == self.pad_to:
            return multitrack_pianoroll
        if input_len > self.pad_to:
            raise RuntimeError(f'Given pad_to {self.pad_to}, but input has length {input_len}')
        padding = np.ones((self.pad_to - input_len, input_width)) * self.pad_token
        
        for track in multitrack_pianoroll.tracks:
            track.pianoroll = np.vstack((track.pianoroll, padding))
        
            
            