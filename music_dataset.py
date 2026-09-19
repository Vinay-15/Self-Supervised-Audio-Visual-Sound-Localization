import h5py
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset


def waveform_to_logmel(waveform, sr=16000, n_mels=128, n_fft=400, hop_length=160):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


class MUSICDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._h5 = None

        with h5py.File(h5_path, 'r') as f:
            self.n_samples = f.attrs['n_samples']
            self.labels = f['labels'][:]
            self.instruments = [
                x.decode() if isinstance(x, bytes) else x
                for x in f['instruments'][:]
            ]

        self.inst_indices = {}
        for i, inst in enumerate(self.instruments):
            self.inst_indices.setdefault(inst, []).append(i)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        f = self._open()

        frame = f['frames'][idx]
        waveform_1 = f['waveforms'][idx]
        label_1 = int(self.labels[idx])
        inst_1 = self.instruments[idx]

        other_insts = [k for k in self.inst_indices if k != inst_1]
        if not other_insts:
            other_insts = list(self.inst_indices.keys())
        other_inst = np.random.choice(other_insts)
        other_idx = np.random.choice(self.inst_indices[other_inst])
        waveform_2 = f['waveforms'][other_idx]
        label_2 = int(self.labels[other_idx])

        mixed_waveform = waveform_1 + waveform_2

        mixed_spec = waveform_to_logmel(mixed_waveform)
        solo_spec = waveform_to_logmel(waveform_1)

        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float()
        mixed_t = torch.from_numpy(mixed_spec).unsqueeze(0).float()
        solo_t = torch.from_numpy(solo_spec).unsqueeze(0).float()

        return {
            'frame': frame_t,
            'mixed_spec': mixed_t,
            'solo_spec': solo_t,
            'label': label_1,
            'other_label': label_2,
        }

    def __del__(self):
        if hasattr(self, "_h5") and self._h5 is not None:
            self._h5.close()
