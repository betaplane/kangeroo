import tensorflow as tf
import pandas as pd

def ar(ts):
    feat = ts.iloc[:-1].values
    lab = ts.iloc[1:]
    gr = tf.Graph()
    with gr.as_default():
        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)
        lsq = tf.matrix_solve_ls(tf.reshape(x, (-1, 1)), tf.reshape(y, (-1, 1)))
    with tf.Session(graph=gr) as s:
        a = s.run(lsq, {x: feat, y: lab.values})
    return pd.DataFrame(feat.flatten() * a.item(), index=lab.index, columns=lab.columns)

def concat(long, short):
    for i, series in enumerate(short):
        tf.get_variable()
