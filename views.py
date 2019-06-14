from flask import Flask, render_template, request
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import psycopg2
import gensim
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import re
from timeit import default_timer as time


def mean_wv(x, w2vModel):
    return np.mean([w2vModel[w] for w in x if w in w2vModel.vocab], axis=0)


def clean_input(text):
    remove = ['th', 'rd', 'st', '']
    text = text.strip().lower().split()
    text = [w.translate({ord(k): None for k in string.digits}) for w in text if(w not in ENGLISH_STOP_WORDS and w not in string.punctuation)]
    text = [re.sub(r'\W', '', w).strip() if hasattr(w, 'strip') else re.sub(r'\W', '', w) for w in text]
    text = [w for w in text if (w not in remove)]
    return text


app = Flask(__name__)

start = time()
user = 'tk'  # add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'chewy'
db = create_engine('postgres://%s%s/%s' % (user, host, dbname))
con = None
con = psycopg2.connect(database=dbname, user=user)

df = pd.read_sql_query(
    """
    SELECT title, review, rating, toys.name AS toy, tokens.tokens, product_info.name, product_info.image, product_info.url
    FROM reviews
    JOIN toys ON reviews.toy_id = toys.id
    JOIN tokens ON reviews.review_id = tokens.review_id
    JOIN product_info ON reviews.toy_id = product_info.toy_id
    WHERE product_info.avg_rating >= 4 AND product_info.reviews >= 20
    """,
    con
)

print("Loaded DataFrame in {:.2f} s".format(time() - start))
tmp = time()
w2vModel = gensim.models.KeyedVectors.load("static/data/w2v_model")
review_wv = np.vstack(df.tokens.apply(mean_wv, w2vModel=w2vModel.wv).values)
print("Reviews word2vec in {:.2f} s ({:.2f} s total)".format(time() - tmp, time() - start))


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/recommender")
def recommender():
    return render_template("recommender.html")


@app.route('/toys')
def recommendations(df=df, model=w2vModel):
    text = request.args.get("user_input")
    text = clean_input(text)
    user_wv = mean_wv(text, w2vModel=model.wv)
    df['sims'] = cosine_similarity(user_wv.reshape(1, -1), review_wv)[0]
    tmp = df.groupby('toy').sims.mean().nlargest(10)
    results = pd.merge(tmp, df, on='toy').drop_duplicates(subset='toy')
    toys = []
    base_url = "https://www.chewy.com/"
    for i, row in results.iterrows():
        toys.append(dict(sim=row['sims_x'],
                         name=row['name'],
                         image="https://" + row['image'],
                         url=base_url + row['url']))
    return render_template("toys.html", toys=toys)


if __name__ == "__main__":
    app.run(debug=True)
