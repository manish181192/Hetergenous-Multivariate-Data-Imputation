
class Config(object):

    verbose_ouput = False

    DATA_PATH = "/home/manish/ML/Rutgers/Project/ML3/ML3/ML3AllSites.csv"
    SAVE_DIR = "SavedModel"
    model_name = "model.ckpt"


    FEATURE_JSON = "features_backup3.json"
    vocab_dir = "DataAnalysis/vocab_dir/"
    reverse_vocab_dir = "DataAnalysis/reverse_vocab_dir/"
    restore_model = False
    save_model = True
    save_gen_samples = True
    num_samples = 1
    display_samples = True
    num_features = None #updated by DataManager

    test_percent = 0.3
    num_epochs = 250
    test_epochs = 10
    num_experiments = 2
    use_kfold_cv = False

    batch_size = None
    lr = 0.005
    s_hp = 0.
    z_hp = 0.0

    s_dim = 10
    z_dim =10
    y_dim = 50

    s_num_layers = []
    z_num_layers = [50, 100, 150]
    y_num_layers = [50, 100, 150]
    xDecoder_num_layers = [100, 100, 100]

    # NLP
    NLP_PREPROCESSED_DATA = "DataAnalysis/nlp_processed_data/"
    nlp=False
    pretrained = False
    word_emb_size = 10
    lstm_hidden_size = 10 #equal to y_dim

    #Multilabel
    MULTILABEL_DIR = "DataAnalysis/multilabel_processed_data/"
