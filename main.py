import telebot
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Chatbot:
    def __init__(self, api_key):
        self.bot = telebot.TeleBot(api_key)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def start(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            self.bot.reply_to(
                message, "Hello! I am an AI-powered chatbot. How can I assist you?")

        @self.bot.message_handler(func=lambda message: True)
        def generate_text(message):
            # Preprocess the user input
            input_text = message.text
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

            # Generate response using the GPT-2 model
            output = self.model.generate(
                input_ids, max_length=100, temperature=0.8, num_return_sequences=1)
            generated_text = self.tokenizer.decode(
                output[0], skip_special_tokens=True)

            # Send the generated response back to the user
            self.bot.reply_to(message, generated_text)

        # Start listening to incoming messages
        self.bot.polling()


if __name__ == "__main__":
    api_key = "YOUR_API_KEY"  # Replace with your API Key
    chatbot = Chatbot(api_key)
    chatbot.start()
