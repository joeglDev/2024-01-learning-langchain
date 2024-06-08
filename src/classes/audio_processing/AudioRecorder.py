import pyaudio
import math
import struct
import wave
import time
import os


# Note: needs sudo apt install portaudio19-dev
class AudioRecorder:
    def __init__(self):
        self.path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/../../data/input_audio/input.wav"
        )
        self.CONTINUE = True
        self.threshold = 100
        self.short_normalize = 1.0 / 32768.0
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.swidth = 2
        self.timeout_length = 3
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk,
        )

    def _rms(self, frame) -> float:
        count = len(frame) / self.swidth
        format = "%dh" % count
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * self.short_normalize
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)
        value = rms * 1000

        return value

    def _record(self):
        print("Noise detected, recording beginning")
        rec = []
        current = time.time()
        end = time.time() + self.timeout_length

        while current <= end:

            data = self.stream.read(self.chunk)
            if self._rms(data) >= self.threshold:
                end = time.time() + self.timeout_length

            current = time.time()
            rec.append(data)
        self._write(b"".join(rec))

    def _write(self, recording):
        wf = wave.open(self.path, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(recording)
        wf.close()

    def listen(self):
        print("Listening...")
        while self.CONTINUE:
            input = self.stream.read(self.chunk)
            rms_val = self._rms(input)
            # print('Current noise', rms_val)
            if rms_val > self.threshold:
                self._record()
                self.CONTINUE = False
