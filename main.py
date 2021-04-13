import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

def createMLModel():
    df=pd.read_csv('datasetRCV.csv')

    y = df['target']
    X = df.drop('target',axis=1)

    numerical_features = ['age','avg_glucose_level','bmi']
    categorical_features = ['gender', 'ever_married', 'work_type','Residence_type','smoking_status','hypertension','heart_disease']

    numerical_pipeline = make_pipeline(SimpleImputer(strategy = 'mean'),StandardScaler())
    categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'),OneHotEncoder())

    preprocessor = make_column_transformer((numerical_pipeline,numerical_features),(categorical_pipeline,categorical_features))

    y_hat_tuned=make_pipeline(preprocessor,LogisticRegression(C=0.05,penalty='l1',solver='liblinear',max_iter=100,random_state=0)) 
    y_hat_tuned.fit(X,y)
    print("Model ready")

    return y_hat_tuned

def main():
    updater = Updater(bot_token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text, get_news_with_keyword))
    dp.add_handler(CallbackQueryHandler(buttonHandler))
    updater.start_polling()
    updater.idle()



if __name__ == '__main__':
    createMLModel()
