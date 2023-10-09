#Libraries
from . import config
import tensorflow as tf
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def rdb_block(ip,no_layers):
    channel = ip.get_shape()[-1] #last value is the number of channels
    st_op = [ip]
    for _ in range(no_layers):
        concat=tf.concat(st_op,axis=-1)
        out = Conv2D(filters=channel,kernel_size=3,padding="same",activation="relu",
                     kernel_initializer="Orthogonal")(concat)
        st_op.append(out)
    final_concat=tf.concat(st_op,axis=-1)
    main_out=Conv2D(filters=ip.get_shape()[-1],kernel_size=1,padding="same",
                    activation="relu",kernel_initializer="Orthogonal")(final_concat)
    main_out=Add()([main_out,ip])
    return main_out

def fetch_subpixel_net(downsample=config.downsample_factor,channels=1,
                 rdb_layer=config.rdb_layer):
    ip = Input((None,None,1))
    x = Conv2D(64, 5, padding="same", activation="relu",kernel_initializer="Orthogonal")(ip)
    x = Conv2D(64, 3, padding="same", activation="relu",kernel_initializer="Orthogonal")(x)
    x = rdb_block(x, numLayers=rdb_layer)
    x = Conv2D(32, 3, padding="same", activation="relu",kernel_initializer="Orthogonal")(x)
    x = rdb_block(x, numLayers=rdb_layer)
    x = Conv2D(channels * (downsample ** 2), 3, padding="same",activation="relu", 
            kernel_initializer="Orthogonal")(x)
    op = tf.nn.depth_to_space(x, downsample) #rearrange data from depth to spatial data
    model = Model(ip, op)
    return model        

