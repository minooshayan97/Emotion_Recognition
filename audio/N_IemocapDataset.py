import os
import torch
import torchaudio
import pandas as pd
import numpy as np

class IemocapDataset(object):
    """
        Create a Dataset for Iemocap. Each item is a tuple of the form:
        (waveform, sample_rate, emotion, activation, valence, dominance)
    """

    _ext_audio = '.wav'
    #_emotions = { 'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8, 'dis': 9, 'oth':10 }

    def __init__(self,
                 root='IEMOCAP_full_release',
                 emotions=['ang', 'hap', 'exc', 'sad', 'fru', 'fea', 'sur', 'neu', 'xxx', 'dis', 'oth'],
                 sessions=[1, 2, 3, 4, 5],
                 script_impro=['script', 'impro'],
                 genders=['M', 'F'],
                 emotion_dict = { 'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8, 'dis': 9, 'oth':10 }):
        """
        Args:
            root (string): Directory containing the Session folders
        """
        self.root = root
        self._emotions = emotion_dict
        # Iterate through all 5 sessions
        data = []
        for i in range(1, 6):
            # Define path to evaluation files of this session
            path = os.path.join(root, 'Session' + str(i), 'dialog', 'EmoEvaluation')

            # Get list of evaluation files
            files = [file for file in os.listdir(path) if file.endswith('.txt')]

            # Iterate through evaluation files to get utterance-level data
            for file in files:
                # Open file
                f = open(os.path.join(path, file), 'r')

                # Get list of lines containing utterance-level data. Trim and split each line into individual string elements.
                data += [line.strip()
                             .replace('[', '')
                             .replace(']', '')
                             .replace(' - ', '\t')
                             .replace(', ', '\t')
                             .split('\t')
                         for line in f if line.startswith('[')]

        # Get session number, script/impro, speaker gender, utterance number
        data = [d + [d[2][4], d[2].split('_')[1], d[2][-4], d[2][-3:]] for d in data]

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['start', 'end', 'file', 'emotion', 'activation', 'valence', 'dominance', 'session', 'script_impro', 'gender', 'utterance'], dtype=np.float32)

        # Filter by emotions
        filtered_emotions = self.df['emotion'].isin(emotions)
        self.df = self.df[filtered_emotions]

        # Filter by sessions
        filtered_sessions = self.df['session'].isin(sessions)
        self.df = self.df[filtered_sessions]

        # Filter by script_impro
        filtered_script_impro = self.df['script_impro'].str.contains('|'.join(script_impro))
        self.df = self.df[filtered_script_impro]

        # Filter by gender
        filtered_genders = self.df['gender'].isin(genders)
        self.df = self.df[filtered_genders]

        # Reset indices
        self.df = self.df.reset_index()

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.float32)
        
        # Map file to correct path w.r.t to root
        self.df['file'] = [os.path.join('Session' + file[4], 'sentences', 'wav', file[:-5], file + self._ext_audio) for file in self.df['file']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        emotion = self.df.loc[idx, 'emotion']
        activation = self.df.loc[idx, 'activation']
        valence = self.df.loc[idx, 'valence']
        dominance = self.df.loc[idx, 'dominance']

        sample = {
            'path': audio_name,
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion,
            'activation': activation,
            'valence': valence,
            'dominance': dominance
        }

        return sample
    