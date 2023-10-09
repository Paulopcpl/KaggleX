#libraries 
from . import config
import tensorflow as tf

def input_processing(img_path,downsample=config.downsample_factor):
    resize_shape=config.org_size[0] // downsample
    print(img_path)
    org_img=tf.io.read_file(img_path)
    org_img=tf.image.decode_jpeg(org_img,3) #number of channels 3
    org_img=tf.image.convert_image_dtype(org_img,tf.float32)
    org_img=tf.image.resize(org_img,config.org_size,method="area")
    org_img_yuv=tf.image.rgb_to_yuv(org_img) #human eye more susceptible to brighntess as compared to hue
    #https://www.sensoray.com/support/appnotes/frame_grabber_capture_modes.htm
    (target,_,_)=tf.split(org_img_yuv,3,axis=-1)
    #resize img to lower res
    down_img=tf.image.resize(target,[resize_shape,resize_shape],method="area")
    target=tf.clip_by_value(target,0.0,1.0)
    down_img=tf.clip_by_value(down_img,0.0,1.0) #clip the values in [0,1]
    return (down_img,target)


