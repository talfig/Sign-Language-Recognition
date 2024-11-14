# model/language_model.py

import os
import openai
import re
import pyttsx3


class LanguageModelAssistant:
    def __init__(self, model_name="gpt-4o-mini", max_tokens=50):
        # Load API key from environment variable
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        openai.api_key = self.api_key
        self.model_name = model_name
        self.max_tokens = max_tokens

        # Initialize the text-to-speech engine
        self.tts_engine = pyttsx3.init()

    def generate_response(self, prompt):
        """
        Generates a response from the language model based on the given prompt.
        Returns the response as a string or an error message if an exception occurs.
        """
        try:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_tokens
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"

    @staticmethod
    def tokenize_text(text):
        """
        Tokenizes the input text into words. If spaces are missing, attempts to identify individual words.
        """
        # If the text already has spaces, split by space
        if ' ' in text:
            tokens = text.split()
        else:
            # Attempt to split based on common patterns and word length
            tokens = re.findall(r'[A-Z][^A-Z]*', text)  # Example heuristic for CamelCase-style
            if not tokens:  # Fallback if no capitalized pattern is found
                tokens = [text[i:i + 4] for i in range(0, len(text), 4)]  # Split into smaller chunks

        return tokens

    def build_sentence_from_tokens(self, tokens):
        """
        Builds a coherent sentence from a list of tokens using the language model.
        """
        token_str = ' '.join(tokens)
        prompt = f"Reformulate the following words into a coherent sentence: {token_str}"
        return self.generate_response(prompt)

    def build_sentence_from_user_input(self, text):
        """
        Takes a potentially unstructured input from the user, tokenizes it, and builds a coherent sentence.
        """
        tokens = self.tokenize_text(text)
        return self.build_sentence_from_tokens(tokens)

    def speak_sentence(self, sentence):
        """
        Converts the generated sentence into speech and speaks it aloud using pyttsx3.
        """
        self.tts_engine.say(sentence)
        self.tts_engine.runAndWait()


# Example usage
if __name__ == "__main__":
    assistant = LanguageModelAssistant()

    # Example text with missing spaces
    text_input = "thisisatestwithnospaces"

    # Generate a sentence from the input
    sentence = assistant.build_sentence_from_user_input(text_input)
    print("Generated Sentence:", sentence)

    # Speak the sentence
    assistant.speak_sentence(sentence)