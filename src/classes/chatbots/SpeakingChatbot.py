import torch
from TTS.api import TTS

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from playsound import playsound

from src.classes.chatbots.chatbot import Chatbot


class SpeakingChatbot(Chatbot):
    def __init__(self, model: Ollama, system: ChatPromptTemplate):
        super().__init__(model, system)

    def speak_completion(self, completion: str):
        # todo convert this funct into a dedicated TTS class handler

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Init TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        # Run TTS
        # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
        # Text to speech list of amplitude values as output
        print("Generating voice")
        wav = tts.tts(text=completion, speaker="Ana Florence", language="en")
        # todo convert list of amplitude values into .wav and play audio
        # Play the generated audio file
        # Generate speech by cloning a voice using default settings
        wav = tts.tts_to_file(
            text=completion,
            file_path="./output.wav",
            speaker="Ana Florence",
            language="en",
            split_sentences=True,
        )

        # play audio from file
        playsound("./output.wav")

    def get_completion(self, prompt: str):
        """Gets a basic chat completion with chat history"""

        print("Thinking...")
        chain = self.System | self.Model
        self.History.add_user_message(prompt)

        model_input = {"messages": self.History.messages}

        completion: str = chain.invoke(model_input)
        self.History.add_ai_message(completion)

        self.speak_completion(completion)

        return completion
