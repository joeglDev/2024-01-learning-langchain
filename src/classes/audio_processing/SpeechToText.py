import os

from torch import Tensor
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.classes.audio_processing.AudioRecorder import AudioRecorder


# todo: microphone input lasts as long as user is speaking
# todo: better audio model for speech to text or spelling correction
# todo: function to clean up input audio
class SpeechToText:
    def __init__(self):
        self.path = (
                os.path.dirname(os.path.abspath(__file__)) + "/../../data/input_audio/input.wav"
        )

    def _record_audio_to_file(self):
        a = AudioRecorder()
        a.listen()


    def _load_audio_from_file(self) -> tuple[Tensor, int]:
        waveform, sample_rate = torchaudio.load(self.path)
        print(f"Original sample rate: {sample_rate}")
        return waveform, sample_rate

    def _resample_audio(self, waveform: Tensor, original_sample_rate: int) -> tuple[Tensor, int]:
        new_sample_rate = 16000
        print(f"Resampling audio to {new_sample_rate}hz")

        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        waveform = resampler(waveform)
        return waveform, new_sample_rate

    def _preprocess_audio(self, waveform: Tensor, sample_rate: int):
        print('Preprocessing audio')
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate)
        mfcc_transform(waveform)

    def _transcribe_audio(self, resampled_waveform: Tensor) -> str:
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
        self._record_audio_to_file()

        waveform, sample_rate = self._load_audio_from_file()
        resampled_waveform, new_sample_rate = self._resample_audio(waveform, sample_rate)

        self._preprocess_audio(resampled_waveform, new_sample_rate)

        transcript = self._transcribe_audio(resampled_waveform)
        first_transcript = transcript[0]

        print('Transcript: ', first_transcript)
        return first_transcript



