from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler, Filters, CallbackQueryHandler, ConversationHandler
# from helper import empty_value, is_channel
# from config.db import DB
# from ml.nltk1 import tes1
from ml.nltk1 import DitaAjaNLTK
import requests
from datetime import datetime, timedelta

NLP = DitaAjaNLTK()

def message(update, context):
    bot = context.bot
    message = update.message.text
    response = NLP.getResponse(message)
    if response == '' or response is None:
        response = "TES"
    bot.send_message(text=response, chat_id=update.message.chat.id)


def handler():
    NLP.nltk_init()
    handlers = [
        MessageHandler(Filters.text, message)
        # MessageHandler(Filters.regex(r'^\[DONITA-ALERT\]'), get_alarm),
        # MessageHandler(Filters.text, no_access)
    ]
    return handlers