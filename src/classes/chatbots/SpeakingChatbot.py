from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

from src.classes.audio_processing.SpeechSynthesis import SpeechSynthesis
from src.classes.audio_processing.SpeechToText import SpeechToText
from src.classes.chatbots.chatbot import Chatbot


# todo: run loop where runs continueally looking for audio
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

    def run(self):
        print("Type 'y' and speak to chat.")
        print("Type 'n' to end the chat.")

        while self.continue_chat:
            user_keystroke = input("Continue chatting?")
            if user_keystroke.strip().lower() == "y":
                handle_speech_to_text = SpeechToText()
                prompt = handle_speech_to_text.run()
                completion = self.get_completion(prompt)
                print(f"Output: {completion}")

            else:
                self.continue_chat = False
                log = self.get_history()
                print("The chat has ended.")
                print("Chat log:")
                print(f"Length of chat: {len(log)}")
                print(log)

