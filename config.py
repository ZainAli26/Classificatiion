
Resnet = {
    "num_layers_freeze": 7
}

TrainingParameters = {
    "train_val_split" : 30,
    "learning_rate": 1e-2,
    "momentum": 0.5,
    "batch_size": 4,
    "num_workers": 0
}

DataPaths = {
    "save_model_path": "./models/",
    "data_path": "../ImagesData/train",
    "barcode_file_path": "/home/smartcart/Desktop/Zain/CropsExtractionDepth/Data/barcodes.txt",
}