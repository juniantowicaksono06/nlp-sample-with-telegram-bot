import traceback
from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler, Filters, CallbackQueryHandler, ConversationHandler
# from helper import empty_value, is_channel
# from config.db import DB
# from ml.nltk1 import tes1
# from ml.nltk1 import DitaAjaNLTK
from ml.nlp import DitaAjaNLP
import requests
from datetime import datetime, timedelta
from os import path
import sys
from telegram import CallbackQuery, File,InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove, ParseMode
import telegram
import base64

# NLP = DitaAjaNLTK()

# def message(update, context):
#     bot = context.bot
#     message = update.message.text
#     response = NLP.getResponse(message)
#     if response == '' or response is None:
#         return
#     # bot.send_message(text=response, chat_id=update.message.chat.id)
#     tag = response['tag']
#     message = response['response']
#     if path.isfile('./assets/img/{}.jpg'.format(tag)):
#         print(tag)
#         with open('./assets/img/{}.jpg'.format(tag), mode='rb') as file:
#             photo = file.read()
#             bot.send_photo(update.message.chat.id, photo=photo, caption=message)
#     else:
#         bot.send_message(text=message, chat_id=update.message.chat.id)
def messagev2(update, context):
    NLP = DitaAjaNLP()
    bot = context.bot
    message = update.message.text
    result = NLP.getTextTag(message)
    all_data = {
        "article_title": [],
        "article_id": [],
        "article_tag": [],
    }
    # print(result, flush=True)
    # bot.send_message(text=result, chat_id=update.message.chat.id)
    if(len(result) > 0):
        for text in result:
            res = find_desmita_data(text)
            if res == False:
                continue
            if len(res['article_id']) > 0:
                if res['article_title'] not in all_data['article_title']:
                    all_data["article_title"].extend(res['article_title'])
                if res['article_id'] not in all_data['article_id']:
                    all_data["article_id"].extend(res['article_id'])
                    all_data['article_tag'].extend(res['article_tag'])
        NLP.tensorflow_train(all_data)
        result = NLP.getResponse(message)
        if isinstance(result, dict):
            image = get_image(result['articleID'])
            bot.send_message(text="Berikut article yang anda cari", chat_id=update.message.chat.id)
            bot.send_photo(update.message.chat.id, image)

        # keyboard = generate_btn(all_data)
        # if keyboard != False:
        #     bot.send_message(text="Berikut hasil pencarian yang ditemukan", chat_id=update.message.chat.id, reply_markup=keyboard)
        # else:
        #     bot.send_message(text="Maaf saya tidak menemukan hasil yang dicari", chat_id=update.message.chat.id)
    else:
        bot.send_message(text="Maaf saya tidak menemukan hasil yang dicari", chat_id=update.message.chat.id) 
    # print(all_data, flush=True)
    # NLP.tensorflow_train(all_data)
    # text = NLP.getResponse(message)
    # print("HASIL", flush=True)
    # print(text, flush=True)

def get_image(article_id):
    # base64Image = str(result[0]['article_capture_content'])
    url = "https://desmita.telkomsel.co.id/api"
    params = {'need': "getArticleImage", 'articleId': article_id}
    proxies = {"http": "", "https": ""}
    response = requests.get(url, params=params, timeout=420, proxies=proxies, verify=False)
    result = response.json()
    base64Image = str(result[0]['article_capture_content'])
    if base64Image != None:
        return base64.b64decode(base64Image)
        # return image
    # if base64Image == "None":
    #     text = msg.getMessage(80)
    #     app.editMessage(chatid, messageID, text)
    #     else:
    #     text = msg.getMessage(81)
    #     app.editMessage(chatid, messageID, text)
    #     image = 'article.png'
    #     with open(image, "wb") as file_image:
    #         file_image.write(base64Image.decode('base64'))
    #     app.sendImage(chatid, image)
    #     text = msg.getMessage(129)
    #     app.sendMessage(user.chatid, text)

def generate_btn(data):
    if len(data['article_id']) == 0:
        return False
    keyboard = []
    for (index, article_id) in enumerate(data['article_id']):
        keyboard.append([InlineKeyboardButton(data['article_title'][index], callback_data=article_id)])
    keyboard = InlineKeyboardMarkup(keyboard)
    return keyboard


def find_desmita_data(text):
    try: 
        url = "https://desmita.telkomsel.co.id/api"
        params = {'need': "getSearchArticlev2", 'searchTerm': text}
        proxies = {"http": "", "https": ""}
        response = requests.get(url, params=params, timeout=420, proxies=proxies, verify=False)
        article_id = []
        article_title = []
        article_tag = []
        result = response.json()
        if(len(result) > 0):
            for article in result:
                article_id.append(article['article_id'])
                article_title.append(article['article_title'].lower().strip())
                tags = article['article_tag'].lower().split(', ')
                article_tag.append(tags)
        return {
            "article_id": article_id,
            "article_tag": article_tag,
            "article_title": article_title
        }   
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        return False

    
    


def handler():
    # NLP.nltk_init()
    # NLP.get_model_data()
    handlers = [
        MessageHandler(Filters.text, messagev2)
        # MessageHandler(Filters.regex(r'^\[DONITA-ALERT\]'), get_alarm),
        # MessageHandler(Filters.text, no_access)
    ]
    return handlers