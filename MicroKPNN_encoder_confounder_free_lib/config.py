config = {
    ### Colorectal cancer data with gender as confounder (we also can use age and bmi as confounder as well) (coming from GMrepo database)
    # "data": {
    #     "train_abundance_path": "dataset/CRC_data/crc_abundance_PRJEB6070.csv",
    #     "train_metadata_path": "dataset/CRC_data/crc_metadata_PRJEB6070.csv",
    #     "test_abundance_path": "dataset/CRC_data/crc_abundance_PRJNA397219.csv",
    #     "test_metadata_path": "dataset/CRC_data/crc_metadata_PRJNA397219.csv",
    #     "disease_column": "disease",
    #     "confounder_column": "sex"
    # },

    ### Colorectal cancer data with gender as confounder (we also can use age and bmi as confounder as well) (coming from GMrepo database) with new test
    # "data": {
    #     "train_abundance_path": "dataset/CRC_data/new_crc_abundance_PRJEB6070.csv",
    #     "train_metadata_path": "dataset/CRC_data/crc_metadata_PRJEB6070.csv",
    #     "test_abundance_path": "dataset/CRC_data/crc_abundance_PRJEB27928.csv",
    #     "test_metadata_path": "dataset/CRC_data/crc_metadata_PRJEB27928.csv",
    #     "disease_column": "disease",
    #     "confounder_column": "sex"
    # },

    # ### IBS data with gender as confounder (coming from GMrepo database)
    # "data": {
    #     "train_abundance_path": "dataset/IBS_data/new_train_filtered.csv",
    #     "train_metadata_path": "dataset/IBS_data/train_metadata.csv",
    #     "test_abundance_path": "dataset/IBS_data/new_test_filtered.csv",
    #     "test_metadata_path": "dataset/IBS_data/test_metadata.csv",
    #     "disease_column": "disease",
    #     "confounder_column": "sex"
    # },

    ### T2D data with metformine as confounder (coming frm Metacardis dataset)
    "data": {
        "train_abundance_path": "dataset/MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv",
        "train_metadata_path": "dataset/MetaCardis_data/train_T2D_metadata.csv",
        "test_abundance_path": "dataset/MetaCardis_data/new_test_T2D_abundance_with_taxon_ids.csv",
        "test_metadata_path": "dataset/MetaCardis_data/test_T2D_metadata.csv",
        "disease_column": "PATGROUPFINAL_C",
        "confounder_column": "METFORMIN_C"
    },

    # which level to use when building taxonomy edges (0=superkingdom, …, 5=genus)
    "taxonomy_level": 5,
    
    "training": {
        "num_epochs": 150,
        "batch_size": 256,
        "learning_rate": 0.0002,             # For disease classifier optimizer
        "encoder_lr": 0.002,                 # For encoder (e.g., for distillation phase)
        "classifier_lr": 0.002,              # For confounder classifier (e.g., 'drug' branch)
        "weight_decay": 0, #1e-4,
        "device": "cuda:0"                   # Change to "cpu" if GPU is unavailable
    },
    "model": {
        "latent_dim": 64,                    # Dimension of the latent space
        "num_encoder_layers": 3,             # Number of layers in the encoder (beyond initial projection)
        "num_classifier_layers": 2,          # Number of layers in each classifier branch
        "dropout_rate": 0.0,                 # Dropout probability (set to 0 to disable)
        "norm": "layer",                     # Normalization type ("batch" or "layer")
        "classifier_hidden_dims": [],        # Optional list; if empty, layers are created via halving
        "activation": "relu",                 # Activation function: options (e.g., "relu", "tanh", "leaky_relu")
        "last_activation": "tanh"
    },
    "tuning": {
        # (Optional) Define search spaces for hyperparameter optimization.
        # "num_encoder_layers": [1, 2, 3],
        # "num_classifier_layers": [1, 2, 3],
        # "dropout_rate": [0.0],
        "learning_rate": [1e-5, 
                          2e-5,
                          5e-5,
                          1e-4, 
                          2e-4, 
                          5e-4, 
                          1e-3,
                          2e-3,
                          5e-3,
                          ],
        "encoder_lr": [1e-5, 
                          2e-5,
                          5e-5,
                          1e-4, 
                          2e-4, 
                          5e-4, 
                          1e-3,
                          2e-3,
                          5e-3,
                          ],
        "classifier_lr": [1e-5, 
                          2e-5,
                          5e-5,
                          1e-4, 
                          2e-4, 
                          5e-4, 
                          1e-3,
                          2e-3,
                          5e-3,
                          ],
        # "activation": ["relu", "tanh", "leaky_relu"],
        # "last_activation": ["relu", "tanh", "leaky_relu"],
        "latent_dim": [64],
        # "batch_size": [64],
        # "norm": ["batch", "layer"]
    }, 

    ### pre-training 
    "pretrain_data": {
        "train_abundance_path": "dataset/pretrain_CRC_data/combined_abundance.csv",
        "train_metadata_path": "dataset/pretrain_CRC_data/combined_metadata.csv",
        "test_abundance_path": "dataset/pretrain_CRC_data/combined_abundance.csv",
        "test_metadata_path": "dataset/pretrain_CRC_data/combined_metadata.csv",
        "disease_column": "disease",
        "confounder_column": "sex", 
        "threshold_feature_sum": 20, # For filtering low-abundance samples
    }, 
    "pretrain_training": {
        "num_epochs": 400,
        "batch_size": 64,
        "learning_rate": 0.0001,             # For reconstruction loss optimizer
        "encoder_lr": 0.002,                  # For encoder (e.g., for distillation phase)
        "classifier_lr": 0.002,              # For confounder classifier (e.g., 'drug' branch)
        "weight_decay": 0, #1e-4,
        "device": "cuda:0"                    # Change to "cpu" if GPU is unavailable
    }, 
    # Using the same model config as the main training

    ### fine-tuning
    "finetuning_training": {
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 2e-05,             # For disease classifier optimizer
        "encoder_lr": 5e-04,                 # For encoder (e.g., for distillation phase)
        "classifier_lr": 5e-04,              # For confounder classifier (e.g., 'drug' branch)
        "weight_decay": 0, #1e-4,
        "device": "cuda:0"                    # Change to "cpu" if GPU is unavailable
    },
    # Using the same data and model config as the main training
}
