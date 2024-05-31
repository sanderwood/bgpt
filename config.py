# Configuration for generative modelling and classification
TRAIN_FOLDERS = [
                # "wikipedia/train",  
                # "ag_news/train", 
                # "imagenet32/train", 
                # "cifar/train", 
                # "librispeech8K/train", 
                # "speech_commands8K/train", 
                "irishman/train",
                # "cpu_states/train",
                 ]     # Folder containing training data
EVAL_FOLDERS = [
                # "wikipedia/test",  
                # "ag_news/test", 
                # "imagenet32/test", 
                # "cifar/test", 
                # "librispeech8K/test", 
                # "speech_commands8K/test", 
                "irishman/test",
                # "cpu_states/test",
                ]                                               # Folder containing evaluation data
EVAL_SPLIT = 0.01                                                # Split of evaluation data

# Configuration for the paths
PRETRAINED_PATH = "weights-irish.pth"                            # Path to pre-trained weights
WEIGHTS_PATH = "weights-irish.pth"                        # Path to save weights
LOGS_PATH = "logs-irish.txt"                              # Path to save logs

# Configuration for the model
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
BYTE_NUM_LAYERS = 3                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 12                                           # Number of layers in the encoder
HIDDEN_SIZE = 768                                               # Hidden Size

# Configuration for the training
NUM_EPOCHS = 32                                                 # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 1e-5                                            # Learning rate for the optimizer
BATCH_SIZE = 8                                                  # Batch size for training
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full batch
LOAD_FROM_CHECKPOINT = True                                    # Whether to load weights from a checkpoint
LOAD_FROM_PRETRAINED = False                                     # Whether to load pre-trained weights from a checkpoint
CONVERSION_MODE = None                                          # Mode of conversion None for autoregressive training, 'input->output' for unidirectional conversion, 'input&output' for bidirectional conversion)
WANDB_LOG = True                                                # Whether to log to wandb
SHOW_WARNS = False                                              # Whether to show warnings
DETERMINISTIC = True                                           # Whether to set random seed for reproducibility

# Configuration for inference
INFERENCE_WEIGHTS_PATH = "weights-conversion.pth"               # Path to weights for inference
INPUT_EXT = "abc"                                               # Extension of input files, used for conversion
TARGET_EXT = "mid"                                              # Extension of target files
INPUT_FOLDER = "input"                                          # Folder containing input files
OUTPUT_FOLDER = "output"                                        # Folder to save output files
INFERENCE_MODE = "convert"                                      # Mode of inference (convert or generate)
NUM_SAMPLES = 100                                               # Number of samples to generate (only for generate mode)
TOP_K = 0                                                       # Top k for sampling
TOP_P = 1.                                                      # Top p for sampling
TEMPERATURE = 1                                                 # Temperature for sampling
