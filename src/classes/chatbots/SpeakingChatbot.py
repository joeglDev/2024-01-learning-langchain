from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

from src.classes.SpeechSynthesis import SpeechSynthesis
from src.classes.chatbots.chatbot import Chatbot


class SpeakingChatbot(Chatbot):
    def __init__(self, model: Ollama, system: ChatPromptTemplate):
        super().__init__(model, system)

    def speak_completion(self, completion: str):
        """Speak completion using SpeechSynthesis class"""

        # Baldur Sanjin    Viktor Eka
        tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        speech = SpeechSynthesis(
            tts_model=tts_model, speaker="Baldur Sanjin", language="en"
        )
        speech.run(completion)

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
