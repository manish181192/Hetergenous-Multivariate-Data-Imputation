DATA_PATH = "/home/manish/ML/Rutgers/Project/ML3/ML3/ML3AllSites.csv"
SAVE_DIR = "../SavedModel"
model_name = "model.ckpt"

FEATURE_JSON = "../features.json"
vocab_dir = "../DataAnalysis/vocab_dir/"
reverse_vocab_dir = "../DataAnalysis/reverse_vocab_dir/"
restore_model = True

num_features = None #updated by DataManager

test_percent = 0.3
num_epochs = 10
num_experiments = 2

batch_size = 1024
lr = 0.005

s_dim = 10
z_dim = 10
y_dim = 10