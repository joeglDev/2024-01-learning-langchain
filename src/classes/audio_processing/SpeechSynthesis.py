import os

import torch
from TTS.api import TTS
from playsound import playsound


class SpeechSynthesis:
    def __init__(self, tts_model: str, speaker: str, language: str):
        self.tts_model = tts_model
        self.speaker = speaker
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(self.tts_model).to(self.device)
        self.path = (
            os.path.dirname(os.path.abspath(__file__)) + "/../../data/audio/output.wav"
        )

    def _synthesise_speech(self, completion: str):
        # Run TTS
        # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
        # Text to speech list of amplitude values as output
        print("Generating voice...")
        # wav = self.tts.tts(text=completion, speaker=self.speaker, language=self.language)
        self.tts.tts_to_file(
            text=completion,
            file_path=self.path,
            speaker=self.speaker,
            language=self.language,
            split_sentences=True,
        )

    def _speak(self):
        """play audio from file"""

        print("Speaking...")
        playsound(self.path)

    def _clean_file(self):
        os.remove(self.path)

    def run(self, completion: str):
        self._synthesise_speech(completion)
        self._speak()
        self._clean_file()
