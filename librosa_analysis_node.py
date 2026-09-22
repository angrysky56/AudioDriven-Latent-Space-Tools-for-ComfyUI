import os
import librosa
import numpy as np
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

class LibrosaAnalysisNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {
                    "multiline": False,
                    "default": "path/to/audio/file.wav",
                }),
                "analysis_type": (["default", "onset", "segment", "tempo", "mel", "spectral", "second", "half_second", "beat"], {
                    "default": "default",
                }),
                "window_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 128
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO_ENERGY", "TIMESTAMPS", "STRING", "ANALYSIS_TYPE")
    RETURN_NAMES = ("energy_levels", "timestamps", "analysis_text", "analysis_type")
    FUNCTION = "analyze_audio"
    CATEGORY = "Audio Processing"

    def analyze_audio(self, audio_file, analysis_type, window_size, audio=None):
        try:
            if audio is not None and isinstance(audio, dict) and "waveform" in audio:
                wf = audio["waveform"]
                sr = int(audio.get("sample_rate", 22050))
                if isinstance(wf, torch.Tensor):
                    if wf.dim() == 3:
                        y = wf[0].mean(dim=0).cpu().float().numpy()
                    elif wf.dim() == 2:
                        y = wf.mean(dim=0).cpu().float().numpy()
                    else:
                        y = wf.cpu().float().numpy()
                else:
                    y = np.array(wf, dtype=np.float32)
            else:
                filepath = audio_file
                if folder_paths is not None:
                    if hasattr(folder_paths, "exists_annotated_filepath") and folder_paths.exists_annotated_filepath(filepath):
                        filepath = folder_paths.get_annotated_filepath(filepath)
                    elif not os.path.isfile(filepath):
                        candidate = os.path.join(folder_paths.get_input_directory(), filepath)
                        if os.path.isfile(candidate):
                            filepath = candidate
                if not os.path.isfile(filepath):
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")
                y, sr = librosa.load(filepath, sr=None)

            if y.ndim > 1:
                y = np.mean(y, axis=0)
            y = np.ascontiguousarray(y, dtype=np.float32)
            duration = float(librosa.get_duration(y=y, sr=sr))

            energy_levels = []
            timestamps = []

            if analysis_type == "onset":
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                if len(onset_frames) > 0:
                    timestamps = librosa.frames_to_time(onset_frames, sr=sr).tolist()
                    rms = librosa.feature.rms(y=y, frame_length=window_size)[0]
                    energy_levels = [float(rms[min(f, len(rms)-1)]) for f in onset_frames]
                else:
                    timestamps = [0.0]
                    energy_levels = [0.0]

            elif analysis_type == "segment":
                hop_length = window_size // 4
                mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
                n_segments = min(max(2, int(duration)), mfcc.shape[1])
                bounds = librosa.segment.agglomerative(mfcc, n_segments)
                timestamps = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length).tolist()
                rms = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)[0]
                energy_levels = [float(rms[min(f, len(rms)-1)]) for f in bounds]

            elif analysis_type == "tempo":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo_frames = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
                timestamps = tempo_frames.tolist()
                energy_levels = [float(e) for e in onset_env]

            elif analysis_type == "mel":
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                timestamps = librosa.frames_to_time(range(mel_spec.shape[1]), sr=sr).tolist()
                energy_levels = [float(e) for e in np.mean(mel_spec, axis=0)]

            elif analysis_type == "spectral":
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                timestamps = librosa.frames_to_time(range(spec_cent.shape[1]), sr=sr).tolist()
                energy_levels = [float(e) for e in spec_cent[0]]

            elif analysis_type == "beat":
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                hop_length = window_size // 4
                rms = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)[0]
                if len(beat_frames) > 0:
                    timestamps = librosa.frames_to_time(beat_frames, sr=sr).tolist()
                    energy_levels = [float(rms[min(f, len(rms)-1)]) for f in beat_frames]
                else:
                    # Fallback if no rhythmic beats detected: sample at regular 0.5s intervals
                    target_times = np.arange(0, duration + 1e-4, 0.5) if duration > 0 else np.array([0.0])
                    idx = [int(np.clip(int(t * sr / hop_length), 0, len(rms)-1)) for t in target_times]
                    timestamps = [float(t) for t in target_times]
                    energy_levels = [float(rms[i]) for i in idx]

            elif analysis_type == "second":
                target_times = np.arange(0, duration + 1e-4, 1.0) if duration >= 1.0 else np.array([0.0])
                hop_length = window_size // 4
                rms = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)[0]
                idx = [int(np.clip(int(t * sr / hop_length), 0, len(rms)-1)) for t in target_times]
                timestamps = [float(t) for t in target_times]
                energy_levels = [float(rms[i]) for i in idx]

            elif analysis_type == "half_second":
                target_times = np.arange(0, duration + 1e-4, 0.5) if duration >= 0.5 else np.array([0.0])
                hop_length = window_size // 4
                rms = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)[0]
                idx = [int(np.clip(int(t * sr / hop_length), 0, len(rms)-1)) for t in target_times]
                timestamps = [float(t) for t in target_times]
                energy_levels = [float(rms[i]) for i in idx]

            else:  # default
                hop_length = window_size // 4
                energy = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length)[0]
                timestamps = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length).tolist()
                energy_levels = [float(e) for e in energy]

            # Normalize energy levels
            energy_arr = np.nan_to_num(np.array(energy_levels, dtype=np.float64), nan=0.0)
            if len(energy_arr) > 0:
                min_val = float(np.min(energy_arr))
                max_val = float(np.max(energy_arr))
                if max_val > min_val:
                    energy_arr = (energy_arr - min_val) / (max_val - min_val)
                else:
                    energy_arr = np.ones_like(energy_arr) if max_val > 0 else np.zeros_like(energy_arr)
                energy_levels = [float(e) for e in energy_arr]
                timestamps = [float(t) for t in timestamps]
            else:
                energy_levels = [0.0]
                timestamps = [0.0]

            analysis_text = (
                f"Analysis Type: {analysis_type}\n"
                f"Duration: {duration:.2f} seconds\n"
                f"Sample Rate: {sr} Hz\n"
                f"Measurements: {len(energy_levels)}\n"
                f"Window Size: {window_size}"
            )

            return (energy_levels, timestamps, analysis_text, analysis_type)

        except Exception as e:
            return ([1.0], [0.0], f"Error: {str(e)}", "default")

NODE_CLASS_MAPPINGS = {
    "LibrosaAnalysisNode": LibrosaAnalysisNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LibrosaAnalysisNode": "Librosa Audio Analysis"
}
