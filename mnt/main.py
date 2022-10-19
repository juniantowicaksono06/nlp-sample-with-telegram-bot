import os
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler, Filters, CallbackQueryHandler, ConversationHandler
from dotenv import load_dotenv
from list_handler import handler
load_dotenv()

class Bot():
    def __init__(self):
        token = os.environ.get('BOT_TOKEN')
        self.updater = Updater(token=token, use_context=True, request_kwargs={'read_timeout': 1000, 'connect_timeout': 1000})
        self.dispatcher = self.updater.dispatcher
    
    def start(self):
        print("Listening for new message")
        self.updater.start_polling()
    
    def register_handler(self):
        handler_list = handler()
        for handler1 in handler_list:
            self.dispatcher.add_handler(handler1)

nlp_bot = Bot()
nlp_bot.register_handler()
nlp_bot.start()


