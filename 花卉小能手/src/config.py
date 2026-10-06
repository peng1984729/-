import os
import torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")          
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")   
MODEL_DIR = os.path.join(BASE_DIR, "models")           
IMG_SIZE = 224         
BATCH_SIZE = 32        
NUM_WORKERS = 2       
VAL_RATIO = 0.15       
TEST_RATIO = 0.15 
SEED = 214       
LEARNING_RATE = 1e-3     
EPOCHS = 20            
WEIGHT_DECAY = 1e-4    
STEP_SIZE = 7          
GAMMA = 0.1         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for _dir in (RAW_DIR, DATA_DIR, MODEL_DIR):
    os.makedirs(_dir, exist_ok=True)
