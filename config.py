DATA_PATH = "/home/manish/ML/Rutgers/Project/ML3/ML3/ML3AllSites.csv"
SAVE_PATH = "/home/manish/SypderProjects/MachineLearningAssignments/MLProject/SavedModel"
model_name = "model.ckpt"

features = ['Site', 'Age'] #features to select(None to select all)
num_features = None #updated by DataManager

test_percent = 0.3
num_epochs = 10
num_experiments = 2

batch_size = 1024
lr = 0.005