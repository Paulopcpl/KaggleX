#Libraries
import os

#Paths
root = os.path.join("BSR","BSDS500","data","images")
train = os.path.join(root, "train")
val = os.path.join(root, "val")
test = os.path.join(root, "test")
superes_model = os.path.join("op", "superes_model")
plot_train = os.path.join("op", "train_plot.png")
viz_path = os.path.join("op","viz")

org_size = (300,300) #given in the docs
downsample_factor = 3

rdb_layer = 3 # No of conv layer in Residual Dense Block (for super res)
batch = 8
epoch = 50
learning_rate = 1e-3


