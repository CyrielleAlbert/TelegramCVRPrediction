import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
import json


AVERAGE_GLUCOSE_LEVEL = 117.5
AVERAGE_BMI= 20
FEATURES_NAMES = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

       
global bot_token, model, questions, answers, n_question, temp_user_data

with open('questions.json', 'r') as qFile:
    questions = json.load(qFile)
with open('buttonsAnswers.json', 'r') as answerFile:
    answers = json.load(answerFile)
with open('tokens.txt', 'r') as tokens_file:
    tokens = json.load(tokens_file)
bot_token = tokens["Bot Token"]

n_question = None

model = None

temp_user_data = [None]*len(questions.keys())


def createMLModel():
    df = pd.read_csv('datasetRCV.csv')

    y = df['target']
    X = df.drop('target', axis=1)

    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'ever_married', 'work_type',
                            'Residence_type', 'smoking_status', 'hypertension', 'heart_disease']

    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'), StandardScaler())
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'))

    preprocessor = make_column_transformer(
        (numerical_pipeline, numerical_features), (categorical_pipeline, categorical_features))

    y_hat_tuned = make_pipeline(preprocessor, LogisticRegression(
        C=0.05, penalty='l1', solver='liblinear', max_iter=100, random_state=0))
    y_hat_tuned.fit(X, y)
    print("Model ready")

    return y_hat_tuned

def init():
    global n_question, temp_user_data
    
    n_question = None
    temp_user_data = [None]*len(questions.keys())

def start(update, context):
    username = update.message.from_user.first_name
    text = 'Hi ' + username + ' ! 😊 \n' \
        'Try /predict to start the form and get to know whether there is a risk you get a CardioVascular issue in the futur. \n' \
        'Just a reminder, this is not a serious test. Your data will be stored only while during the test and deleted after the prediction.'
    context.bot.send_message(chat_id=update.effective_chat.id, text=text)


def predict(update, context):
    global n_question

    print("hello")
    n_question = 1
    ask_question(update, context)


def ask_question(update, context):

    print("ask", n_question)
    text = questions[str(n_question)]
    print(answers.keys(),str(n_question), str(n_question) in answers.keys())
    if str(n_question) in answers.keys():
        buttons = []
        for index, answer in enumerate(answers[str(n_question)]):
            buttons.append([InlineKeyboardButton(
                answer, callback_data=str(index))])
        reply_markup = InlineKeyboardMarkup(buttons)
        context.bot.send_message(chat_id=update.effective_chat.id,text=text, reply_markup=reply_markup)
    else:
        context.bot.send_message(chat_id=update.effective_chat.id,text=text)


def getWrittenAnswers(update, context):
    global temp_user_data

    if str(n_question) not in answers.keys() and not (n_question is None):
        value = update.effective_message.text
        try:
            value = float(value)
            temp_user_data[n_question-1] = value
            print(temp_user_data)
            incrementNQuestion(update, context)
        except:
            if 'null' in value:
                if n_question ==8:
                    temp_user_data[n_question-1] =  AVERAGE_GLUCOSE_LEVEL
                    print(temp_user_data)
                    incrementNQuestion(update, context)
                elif n_question ==9:
                    temp_user_data[n_question-1] =  AVERAGE_BMI
                    print(temp_user_data)
                    incrementNQuestion(update, context)
            else:
                print("An error occured with the value", value)
                update.message.reply_text(
                "I am sorry, I did not understand. Please make sure you send a number and use a a point if you want to add decimals.")      

    elif n_question is None:
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="Write /start to go back to the beginning and /predict to start the prediction")
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id, text="Select one of the button above")


def buttonHandler(update, context):
    global temp_user_data

    if str(n_question) not in answers.keys():
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="Oops an error occured, please start again by writing /predict")
    else:
        query = update.callback_query
        temp_user_data[n_question-1] = int(query.data)
        print(temp_user_data)
        incrementNQuestion(update, context)


def incrementNQuestion(update, context):
    global n_question

    if n_question < 10:
        n_question += 1
        ask_question(update, context)
    else:
        prediction(update,context)

def prediction(update,context):
    text = ['Apparently, you have no risk to get a cardiovascular disease.','From the data we have collected, there is a risk you could get a cardiovascular disease.']
    data = pd.DataFrame(np.array(temp_user_data).reshape(1,-1),columns=FEATURES_NAMES)
    prediction = model.predict(data)
    context.bot.send_message(chat_id=update.effective_chat.id,text=text[prediction[0]])
    context.bot.send_message(chat_id=update.effective_chat.id,text='If you want to try again, write /predict. See you soon!')
    init()

def main():
    updater = Updater(bot_token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('predict', predict))
    dp.add_handler(MessageHandler(Filters.text, getWrittenAnswers))
    dp.add_handler(CallbackQueryHandler(buttonHandler))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    model = createMLModel()
    main()
