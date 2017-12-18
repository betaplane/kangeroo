# import tensorflow as tf
# import pandas as pd
import numpy as np

# def ar(ts):
#     feat = ts.iloc[:-1].values
#     lab = ts.iloc[1:]
#     gr = tf.Graph()
#     with gr.as_default():
#         x = tf.placeholder(tf.float32)
#         y = tf.placeholder(tf.float32)
#         lsq = tf.matrix_solve_ls(tf.reshape(x, (-1, 1)), tf.reshape(y, (-1, 1)))
#     with tf.Session(graph=gr) as s:
#         a = s.run(lsq, {x: feat, y: lab.values})
#     return pd.DataFrame(feat.flatten() * a.item(), index=lab.index, columns=lab.columns)

# def concat(long, short):
#     for i, series in enumerate(short):
#         tf.get_variable()

def ar2(y1, y2):
    z1 = np.zeros_like(y1).flatten()
    z2 = np.ones_like(y2).flatten()
    y = np.hstack((np.asarray(y1).flatten(), np.asarray(y2).flatten()))
    i = np.isfinite(y)
    b = y[i]
    z = np.hstack((z1, z2))[i]
    A = np.vstack((y[:-1], z[:-1])).T
    x = np.linalg.lstsq(A, b[1:])
    return x[0]



if __name__=="__main__":
    x = np.linspace(0, 1, 100)
    f = np.sin(12 * (x + .2)) / (x + .2)
    # f[49:51] = np.nan
    y1 = f[:50]
    y2 = f[50:] + 2
    y = np.hstack((y1, y2))
