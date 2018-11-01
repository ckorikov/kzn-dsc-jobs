"""
Project: kzn-dsc-jobs
"""

import sys
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
import re
import numpy as np
import pickle
import string

from nltk.stem.snowball import EnglishStemmer, RussianStemmer

ru_stemmer = RussianStemmer()
eng_stemmer = EnglishStemmer()

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


models = pickle.load( open( "models.p", "rb" ) )
tfidfs = pickle.load( open( "tfidfs.p", "rb" ) )
# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!:+1:')


def help(bot, update):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(bot, update):
    """Echo the user message."""
    # update.message.reply_text(update.message.text)
    if process_text(update.message.text):
        update.message.reply_text('👎')
    else:
        update.message.reply_text('👍')

def process_text(text):
    formatted_text = perform_transformation(text)
    prob = []
    for model, tfidf in zip(models, tfidfs):
        features = tfidf.transform([formatted_text])
        prob.append(model.predict(features))
        
    return (np.mean(prob) > 0.5)

def perform_transformation(text):
    text = text.lower()
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    words_without_punctuation = text.split()
    # 3. Stem words
    stemmed_words = [eng_stemmer.stem(ru_stemmer.stem(word)) for word in words_without_punctuation]

    return ' '.join(stemmed_words)

def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    token = sys.argv[1] if len(sys.argv) > 1 else None
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(token)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
