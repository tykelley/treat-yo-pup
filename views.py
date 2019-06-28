import re
import string
from timeit import default_timer as time

import json
import numpy as np
import pandas as pd
import psycopg2
from flask import Flask, render_template, request
from gensim.models import KeyedVectors as KV
from gensim.models.phrases import Phraser
from lightfm import LightFM
from multiprocessing import cpu_count
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as ESW
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine


ITEM_ALPHA = 1e-6
NUM_COMPONENTS = 30
NUM_EPOCHS = 10
NUM_THREADS = cpu_count()


def clean_input(text, bigrams, trigrams):
    text = text.strip().lower().split()
    text = [w.translate({ord(k): None for k in string.digits}) for w in text]
    text = [w for w in text if(w not in ESW and w not in string.punctuation)]
    text = [re.sub(r'\W', '', w).strip() if hasattr(w, 'strip') else re.sub(r'\W', '', w) for w in text]
    text = trigrams[bigrams[text]]
    return text


def convert_uim(df, uid_col="user_id", item_col="toy", rating_col="rating", min_rating=None):
    uim = df.groupby([uid_col, item_col])[rating_col].mean().unstack(fill_value=0)
    if min_rating is not None:
        msk = uim.values > min_rating
        uim.values[msk] = 1
        uim.values[~msk] = 0
    return uim


def collab_recommendation(interactions, text, keywords, user_features, adj_map, toy_cols, toy_mapper):
    ua = create_user_vector(text + keywords, adj_map)
    user_features = vstack([user_features, ua]).tocsr()
    n_users, n_toys = interactions.shape

    model = LightFM(loss='warp', item_alpha=ITEM_ALPHA, no_components=NUM_COMPONENTS, random_state=42)
    model.fit(interactions, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
    scores = model.predict(n_users-1, np.arange(n_toys), user_features=user_features)
    inds = np.argsort(scores)[::-1]
    toy_ids = [toy_mapper[toy_cols[i]] for i in inds]
    results = pd.DataFrame({"toy_id": toy_ids, "score": scores[inds]})
    return results


def content_recommendation(text, df, model):
    text = clean_input(text, bigrams, trigrams)
    user_wv = mean_wv(text, w2vModel=model)
    try:
        df['sims'] = cosine_similarity(user_wv.reshape(1, -1), review_wv)[0]
    except ValueError:
        return None
    cols = ['sims', 'name', 'image', 'url', 'toy_id', 'avg_rating', 'reviews', 'price', 'review']
    results = (df.loc[:, cols]
                 .sort_values("sims", ascending=False)
                 .drop_duplicates(subset='toy_id'))
    return results


def create_toy_mapper(df, id_col="toy_id", name_col="toy"):
    return dict(zip(df[name_col].values, df[id_col].values))


def create_user_vector(user_inputs, adj_map):
    user_vector = dict.fromkeys(set(adj_map.values()), 0)

    for col in user_vector:
        if col in user_inputs:
            user_vector[col] += 1
    for col in adj_map:
        if col in user_inputs:
            user_vector[adj_map[col]] += 1

    user_vector = pd.DataFrame([user_vector], index=['user'])
    return user_vector


def mark_toy_cols(uim):
    return {i: k for i, k in enumerate(uim.columns)}


def mean_wv(x, w2vModel):
    return np.mean([w2vModel[w] for w in x if w in w2vModel.vocab], axis=0)


app = Flask(__name__)

with open("./.password") as f:
    password = f.read().strip()

start = time()
user = 'tk'  # add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'chewy'
db = create_engine('postgresql://%s:%s@%s/%s' % (user, password, host, dbname))
con = db.connect()

df = pd.read_sql_query(
    """
    SELECT title, review, rating, reviews.user_id, reviews.toy_id,
           toys.name AS toy, tokens.trigrams AS tokens, product_info.name,
           product_info.image, product_info.url, product_info.avg_rating,
           product_info.reviews, product_info.price
    FROM final_reviews AS reviews
    JOIN toys ON reviews.toy_id = toys.id
    JOIN tokens ON reviews.review_id = tokens.review_id
    JOIN product_info ON reviews.toy_id = product_info.toy_id
    """,
    con
)

bigrams = Phraser.load('static/data/bigram.model')
trigrams = Phraser.load('static/data/trigram.model')
df['tokens'] = (df.tokens.str.split()
                         .apply(lambda x: [i for i in x if i not in ESW]))
print("Loaded DataFrame in {:.2f} s".format(time() - start))
tmp = time()
w2vModel = KV.load_word2vec_format("static/data/w2v_model", binary=True)
review_wv = np.vstack(df.tokens.apply(mean_wv, w2vModel=w2vModel).values)
print("Reviews word2vec in {:.2f} s ({:.2f} s total)".format(time() - tmp,
                                                             time() - start))
tmp = time()
uim = convert_uim(df)
toy_cols = mark_toy_cols(uim)
toy_mapper = create_toy_mapper(df)
user_adjs = csr_matrix(pd.read_csv("static/data/user_adj.csv", index_col=0))
new_user = pd.DataFrame().reindex_like(uim).iloc[:1].fillna(0)
new_user = new_user.rename(index={1: 'user'})
uim = coo_matrix(pd.concat([uim, new_user]))

with open("static/data/adj_mapper.txt") as f:
    adj_map = json.load(f)

print("Loaded UIM data in {:.2f} s ({:.2f} s total)".format(time() - tmp,
                                                            time() - start))


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/recommender")
def recommender():
    return render_template("recommender.html")


@app.route('/toys')
def recommendations(df=df, model=w2vModel, interactions=uim):
    text = request.args.get("user_input")
    keywords = request.args.getlist("user_keywords")
    kw_weight = int(request.args.get("user_kw_weight")) / 6
    content = content_recommendation(text, df, model)
    if content is None:
        return render_template("error.html", user_text=text)
    collab = collab_recommendation(interactions, text, " ".join(keywords),
                                   user_adjs, adj_map, toy_cols, toy_mapper)
    joined = content.merge(collab, on="toy_id", how="left")
    joined['score'] = (joined.score - joined.score.min()) / (joined.score.max() - joined.score.min())
    joined['avg'] = (joined.sims * (1-kw_weight)) + (joined.score * kw_weight)
    joined = joined.sort_values("avg", ascending=False).head(10)
    toys = []
    base_url = "https://www.chewy.com/"
    table_cols = ['avg', 'name', 'image', 'url', 'avg_rating', 'reviews', 'price', 'review']
    for row in joined.loc[:, table_cols].values:
        toys.append(dict(sim=np.round(row[0] * 100, 1),
                         name=row[1],
                         image=row[2],
                         url=base_url + row[3],
                         stars=np.round(row[4], 1),
                         n_reviews=row[5],
                         price=row[6],
                         review=row[7]))
    return render_template("toys.html", toys=toys, user_text=text, user_kw=keywords)


if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
