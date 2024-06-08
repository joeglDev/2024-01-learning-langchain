import os

from torch import Tensor
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import sounddevice as sd
from scipy.io.wavfile import write


class SpeechToText:
    def __init__(self):
        self.path = (
                os.path.dirname(os.path.abspath(__file__)) + "/../../data/input_audio/input.wav"
        )

    def record_audio_to_file(self):
        # todo: end when stop talking not at 5 secs

        # Sampling frequency
        freq = 44100

        # Recording duration
        duration = 5

        # Start recorder with the given values
        # of duration and sample frequency
        print('Recording audio')
        recording = sd.rec(int(duration * freq),
                           samplerate=freq, channels=2)

        # Record audio for the given number of seconds
        sd.wait()
        print('Stopped recording audio')

        # This will convert the NumPy array to an audio
        # file with the given sampling frequency
        write(self.path, freq, recording)

        # Convert the NumPy array to audio file
        #wv.write("input1.wav", recording, freq, sampwidth=2)

    def load_audio_from_file(self) -> tuple[Tensor, int]:
        waveform, sample_rate = torchaudio.load(self.path)
        print(f"Original sample rate: {sample_rate}")
        return waveform, sample_rate

    def resample_audio(self, waveform: Tensor, original_sample_rate: int) -> tuple[Tensor, int]:
        new_sample_rate = 16000
        print(f"Resampling audio to {new_sample_rate}hz")

        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        waveform = resampler(waveform)
        return waveform, new_sample_rate

    def preprocess_audio(self, waveform: Tensor, sample_rate: int):
        print('Preprocessing audio')
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate)
        mfcc_transform(waveform)

    def transcribe_audio(self, resampled_waveform: Tensor) -> str:
        print('Transcribing audio')
        # Load pre-trained model
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        # Perform inference
        with torch.no_grad():
            logits = model(resampled_waveform).logits

        # Decode logits to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = processor.batch_decode(predicted_ids)
        return transcript

    def run(self) -> str:
        self.record_audio_to_file()

        waveform, sample_rate = self.load_audio_from_file()
        resampled_waveform, new_sample_rate = self.resample_audio(waveform, sample_rate)

        self.preprocess_audio(resampled_waveform, new_sample_rate)

        transcript = self.transcribe_audio(resampled_waveform)
        first_transcript = transcript[0]

        print('Transcript: ', first_transcript)
        return first_transcript



pipeline = SpeechToText()
pipeline.run()
