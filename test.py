import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

file_path = "../data/bias_init_vector.npy"

script_dir = os.path.dirname(__file__)
print("obs >",script_dir)
if not(os.path.exists(file_path)):
    print("NOT found!!!")
else:
    print("found***")
