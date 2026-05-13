"""
Configurações globais do projeto.
"""

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 10
NUM_CLASSES = 7
RANDOM_STATE = 42

DATASET_NAME = "kmader/skin-cancer-mnist-ham10000"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
LAYERS_TO_UNFREEZE = 25
DROPOUT_RATE = 0.3

OUTPUT_DIR = "outputs"
MODEL_DIR = "outputs/models"
REPORT_DIR = "outputs/reports"
FIGURE_DIR = "outputs/figures"
