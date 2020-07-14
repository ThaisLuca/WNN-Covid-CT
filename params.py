
#COVIDs Datasets 
COVID_TRAINING_DATASET = '/resources/Data-split/COVID/trainCT_COVID.txt'
COVID_VALIDATION_DATASET = '/resources/Data-split/COVID/valCT_COVID.txt'
COVID_TEST_DATASET = '/resources/Data-split/COVID/testCT_COVID.txt'

#Non-COVIDs Datasets
NON_COVID_TRAINING_DATASET = '/resources/Data-split/NonCOVID/trainCT_NonCOVID.txt'
NON_COVID_VALIDATION_DATASET = '/resources/Data-split/NonCOVID/valCT_NonCOVID.txt'
NON_COVID_TEST_DATASET = '/resources/Data-split/NonCOVID/testCT_NonCOVID.txt'

#Path to COVID and Non-COVID images
COVID_IMAGES_PROCESSED_PATH = '/resources/Images-processed/CT_COVID/'
NON_COVID_IMAGES_PROCESSED_PATH = '/resources/Images-processed/CT_NonCOVID/'

#Path to save pre-processed images
PRE_PROCESSED_OTSU_FOLDER_PATH = '/resources/pre-processed_otsu_threshold'
PRE_PROCESSED_CANNY_FOLDER_PATH = '/resources/pre-processed_canny_edge'

#CSV files
TRAIN_DATASET = '/train_dataset.csv'
VALIDATION_DATASET = '/validation_dataset.csv'
TEST_DATASET = '/test_dataset.csv'

#Binarization Techniques
OTSU_THRESHOLD = 'otsu_threshold'
CANNY_DETECTOR = 'canny_detector'
EMBEDDING_VGG16 = 'embedding_vgg16'
EMBEDDING_VGG19 = 'embedding_vgg19'
EMBEDDING_INCEPTION = 'inception_v3'

# Embeddings CSV

# VGG-16
VGG_16_TRAIN_FEATURES_FILE = 'embeddings/vgg-16-train-features.csv'
VGG_16_TEST_FEATURES_FILE = 'embeddings/vgg-16-test-features.csv'

# VGG-19
VGG_19_TRAIN_FEATURES_FILE = 'embeddings/vgg-19-train-features.csv'
VGG_19_TEST_FEATURES_FILE = 'embeddings/vgg-19-test-features.csv'

# Inception V3
INCEPTION_V3_TRAIN_FEATURES_FILE = 'embeddings/inception-v3-train-features.csv'
INCEPTION_V3_TEST_FEATURES_FILE = 'embeddings/inception-v3-test-features.csv'


VGG_16_TRAINING = 'vgg16_training.csv'
VGG_16_TEST = 'vgg16_test.csv'

VGG_19_TRAINING = 'vgg19_training.csv'
VGG_19_TEST= 'vgg19_test.csv'

INCEPTION_V3_TRAINING = 'inception_training.csv'
INCEPTION_V3_TEST = 'inception_test.csv'

#Images Dimensions
DIM = (300,200)
THRESHOLD = 125
