from superes_v1.data_utils import input_processing
from superes_v1 import config
from superes_v1 import subpixel_net
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf

def peak_signal_noise_ratio(org,pre):
    org=org*255.0
    org=tf.cast(org,tf.uint8)
    org=tf.clip_by_value(org,0,255) #upcaling from [0,1]->[0,255]
    pre=pre*255.0
    pre=tf.cast(pre,tf.uint8)
    pre=tf.clip_by_value(pre,0,255)
    return tf.image.psnr(org,pre,max_val=255)

auto=tf.data.AUTOTUNE
train_path=list(paths.list_images(config.train))
val_path=list(paths.list_images(config.val))
trainds=tf.data.Dataset.from_tensor_slices(train_path)
valds=tf.data.Dataset.from_tensor_slices(val_path)
trainds=trainds.map(input_processing,num_parallel_calls=auto).batch(config.batch).prefetch(auto)
valds=valds.map(input_processing,num_parallel_calls=auto).batch(config.batch).prefetch(auto)
model=subpixel_net.fetch_subpixel_net()
model.compile(optimizer="adam",loss="mse",metrics=peak_signal_noise_ratio)
temp=model.fit(trainds,validation_data=valds,epochs=config.epoch)

#Plots
plt.style.use("ggplot")
plt.figure()
plt.plot(temp.history["loss"], label="train_loss")
plt.plot(temp.history["val_loss"], label="val_loss")
plt.plot(temp.history["psnr"], label="train_psnr")
plt.plot(temp.history["val_psnr"], label="val_psnr")
plt.title("Training Loss and PSNR")
plt.xlabel("Epoch #")
plt.ylabel("Loss/PSNR")
plt.legend(loc="lower left")
plt.savefig(config.plot_train)
model.save(config.superes_model)



