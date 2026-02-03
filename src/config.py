from .models import (
    ModZScoreDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    LSTMDetector,
    AutoencoderDetector
)

"""
Central configuration file for the project.
"""

TRAIN_LEN = 2880  # Number of rows per stream reserved for training (the first 2880)
STREAM_LEN = 3025   # Total number of rows per stream
HOLDOUT_LEN = STREAM_LEN - TRAIN_LEN # Number of rows per stream reserved for holdout (the last 145)

SPECIFIC_INCIDENT_NUMS = [
    9, 13, 16, 14, 17, 20, 23, 50, 52, 53, 54, 55, 56, 60, 63, 67, 68, 69,
    74, 78, 80, 84, 88, 90, 91, 92, 94, 95, 96, 98, 100, 101, 102, 103, 106, 108, 109, 110,
    113, 116, 117, 118, 120, 123, 127, 128, 129, 130, 131, 132, 133, 136, 138, 142,
    145, 151, 152, 153, 154, 155, 159, 161, 162, 166, 170, 171, 172, 173, 174, 176,
    177, 180, 181, 184, 185, 186, 187, 188, 193, 197, 198, 199, 201, 202, 204, 205,
    213, 216, 217, 218, 219, 220, 221, 222, 224, 226, 227, 231, 233, 236, 238, 240, 241,
    242, 243, 245, 251, 254, 255, 259, 262, 263, 264, 265, 267, 270, 271, 273, 274,
    276, 284, 287, 289, 293, 296, 297, 304, 306, 307, 309, 317, 322, 323, 326, 340,
    341, 342, 343, 344, 345, 347, 352, 354, 356, 357, 360, 362, 371, 373, 381, 383,
    387, 390, 393, 394, 395, 398, 401, 403, 405, 407, 408, 410, 414, 415, 416, 420,
    429, 433, 435, 440, 441, 442, 448, 450
]

FEATURE_SETS = {
    'Speed': ['speed_smoothed'],
    'Occupancy': ['occ_smoothed'],
    'Bivariate': ['speed_smoothed', 'occ_smoothed']
}

# --- Model Hyperparameter Grids ---
# This dictionary is the single source of truth for all model configurations and tuning grids.
# The 'params' for each model should be a single dictionary where each key's value is a list of options.

MODEL_CONFIG = {
    'ModZScore': {
        'class': ModZScoreDetector,
        'is_dl': False,
        'params': {
            # No hyperparameters to tune, but the structure must be a dict.
        }
    },
    'IsolationForest': {
        'class': IsolationForestDetector,
        'is_dl': False,
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'max_samples': [0.5, 0.75, 'auto'],
            'random_state': [42] # Keep constant for reproducibility
        }
    },
    'LOF': {
        'class': LOFDetector,
        'is_dl': False,
        'params': {
            'n_neighbors': [10, 20, 35, 50, 75]
        }
    },
    'OneClassSVM': {
        'class': OneClassSVMDetector,
        'is_dl': False,
        'params': {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1],
            'gamma': ['scale', 'auto']
        }
    },
    'LSTM': {
        'class': LSTMDetector,
        'is_dl': True,
        'params': {
            'window_size': [24, 48],
            'lstm_units': [16, 32, 64, 128],
            'n_layers': [1, 2],
            'batch_size': [32, 64],
            'random_state': [42]
            # 'epochs' is removed from tuning and handled by EarlyStopping in the model class
        }
    },
    'Autoencoder': {
        'class': AutoencoderDetector,
        'is_dl': True,
        'params': {
            'window_size': [24, 48],
            'latent_dim': [8, 16],
            'hidden_dim': [64, 128],
            'n_layers': [1, 2],
            'batch_size': [32, 64],
            'random_state': [42],
            'arch': ['dense']
            # 'epochs' is removed from tuning and handled by EarlyStopping in the model class
        }
    },
    'AutoencoderLSTM': {
        'class': AutoencoderDetector,
        'is_dl': True,
        'params': {
            'arch': ['lstm-ae'],
            'window_size': [24, 48],
            'latent_dim': [8, 16],
            'hidden_dim': [32, 64],
            'n_layers': [1, 2],
            'batch_size': [32],
            'random_state': [42],
        }
    }
}
