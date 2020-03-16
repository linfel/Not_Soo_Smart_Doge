import random
import nltk
from final_bot_config import BOT_CONFIG
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Готовим датасет
dataset = []  # [[example, intent], ...]

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        dataset.append([example, intent])

# разделим его на списки примеров и намерений
X_text = [x for x, y in dataset] #examples
y = [y for x, y in dataset] #intents

# Векторизация
# Конвертируем примеры в вектора из чисел.
#
# Можно выбрать любой векторайзер. Также можно произвести обработку текстов заранее (например привести все слова к
# начальной форме).

vectorizer = CountVectorizer(lowercase=True, ngram_range=(3, 3), analyzer='char_wb')
X = vectorizer.fit_transform(X_text)  # вектора примеров

tfift_transformer = TfidfTransformer()
x_tf = tfift_transformer.fit_transform(X)
X = x_tf

clf = LogisticRegression()
clf.fit(X, y)


def get_intent(text):
    vectors = vectorizer.transform([text])
    intent = clf.predict(vectors)[0]

    probas = clf.predict_proba(vectors)[0]
    index = list(clf.classes_).index(intent)
    proba = probas[index]

    if BOT_CONFIG['threshold'] <= proba:
        return intent


chit_chat_dataset = []

with open('dialogues.txt') as dialogues_file:
    content = dialogues_file.read()
    dialogues = content.split('\n\n')
    # print(len(dialogues))
    for dialogue in dialogues:
        replicas = dialogue.split('\n')
        replicas = [replica[2:].strip().lower() for replica in replicas]
        replicas = [replica for replica in replicas if replica]
        for i in range(len(replicas) - 1):
            chit_chat_dataset.append((replicas[i], replicas[i + 1]))
# print(len(chit_chat_dataset))
chit_chat_dataset = list(set(chit_chat_dataset))
# print(len(chit_chat_dataset))

chit_chat_dataset_part = chit_chat_dataset[:BOT_CONFIG['chit_chat_limit']]


def generate_random_answer(text):
    text = text.lower()

    for question, answer in chit_chat_dataset_part:
        if abs(len(text) - len(question)) / len(question) <= (1 - BOT_CONFIG['chit_chat_threshold']):
            # Расстояние Левенштейна
            distance = nltk.edit_distance(text, question)
            # Насколько процентов похожи фразы
            similarity = 1 - min(1, distance / len(question))
            if similarity >= BOT_CONFIG['chit_chat_threshold']:
                return answer


stats = {'rules': 0, 'random_answer': 0, 'failure': 0}


def generate_answer(text):
    # NLU:

    intent = get_intent(text)

    # Answer generation:

    # rules
    if intent is not None:
        responses = BOT_CONFIG['intents'][intent]['responses']
        stats['rules'] += 1
        return random.choice(responses)

    # generative model
    random_answer = generate_random_answer(text)
    if random_answer is not None:
        stats['random_answer'] += 1
        return random_answer

    # return failure phrase
    stats['failure'] += 1
    return random.choice(BOT_CONFIG['failure_phrases'])

"""
Simple Bot to reply to Telegram messages.
First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def generate(update, context):
    """Echo the user message."""
    update.message.reply_text(generate_answer(update.message.text))
    print(stats)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater('904746186:AAFeNyjFHfOLK0mdVLFTirUQHpwOYPZWzpk',
                      request_kwargs={'proxy_url': 'socks5://68.183.25.126:8902'},
                      use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, generate))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


# if __name__ == '__main__':
main()

# def get_intent(text):
#     for intent, intent_data in BOT_CONFIG['intents'].items():
#         for example in intent_data['examples']:
#             # Расстояние Левенштейна
#             distance = nltk.edit_distance(text.lower(), example.lower())
#             # Насколько процентов похожи фразы
#             print(example)
#             print(len(example))
#             similarity = 1 - min(1, distance / len(example))
#             if similarity >= BOT_CONFIG['threshold']:
#                 return intent
