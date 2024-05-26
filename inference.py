import os
import time
import torch
from utils import *
from config import *
from transformers import  GPT2Config

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    hidden_size=HIDDEN_SIZE,
                    n_head=HIDDEN_SIZE//64,
                    vocab_size=1)
byte_config = GPT2Config(num_hidden_layers=BYTE_NUM_LAYERS, 
                    max_length=PATCH_SIZE+1, 
                    max_position_embeddings=PATCH_SIZE+1,
                    hidden_size=HIDDEN_SIZE,
                    n_head=HIDDEN_SIZE//64,
                    vocab_size=256+1)
model = bGPTLMHeadModel(patch_config, byte_config)
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

checkpoint = torch.load(INFERENCE_WEIGHTS_PATH, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

def read_bytes(filename):
    
    ext = filename.split('.')[-1]
    ext = bytearray(ext, 'utf-8')
    ext = [byte for byte in ext][:PATCH_SIZE]

    with open(filename, 'rb') as f:
        file_bytes = f.read()

    bytes = []
    for byte in file_bytes:
        bytes.append(byte)

    if len(bytes)%PATCH_SIZE!=0:
        bytes = bytes + [256] * (PATCH_SIZE - len(bytes) % PATCH_SIZE)

    bos_patch = ext + [256] * (PATCH_SIZE - len(ext))
    bytes = bos_patch + bytes + [256] * PATCH_SIZE
    bytes = bytes[:PATCH_LENGTH*PATCH_SIZE]

    return bytes

bos_patch = [byte for byte in bytearray(TARGET_EXT, 'utf-8')]
bos_patch = bos_patch + [256] * (PATCH_SIZE - len(bos_patch))

if INFERENCE_MODE == "convert":
    files = os.listdir(INPUT_FOLDER)
    files = [i for i in files if i.split('.')[-1] == INPUT_EXT]
else:
    files = list(range(NUM_SAMPLES))

for i in files:
    if INFERENCE_MODE == "convert":
        filename = OUTPUT_FOLDER+"/"+i+'.'+TARGET_EXT
        byte_list = read_bytes(INPUT_FOLDER+"/"+i)[:-PATCH_SIZE]+bos_patch
    else:
        filename = OUTPUT_FOLDER+"/"+time.strftime("%Y%m%d-%H%M%S")+"-"+str(i+1)+"."+TARGET_EXT
        byte_list = bos_patch.copy()
    prefix_len = len(byte_list)
    input_patches = torch.tensor([byte_list], device=device)
    while input_patches.shape[1]<PATCH_LENGTH*PATCH_SIZE:
        predicted_patch = model.generate(input_patches.unsqueeze(0),
                                         top_k=TOP_K,
                                         top_p=TOP_P,
                                         temperature=TEMPERATURE)
        for byte in predicted_patch:
            if byte == 256:
                break
            byte_list.append(byte)
        if byte == 256:
            break
        predicted_patch = torch.tensor([predicted_patch], device=device)
        input_patches = torch.cat([input_patches, predicted_patch], dim=1)

    byte_list = byte_list[prefix_len:]
    # set output file name as the current time
    with open(filename, 'wb') as file:
        for byte in byte_list:
            file.write(bytes([byte]))
        if INFERENCE_MODE == "convert":
            print("Converted to "+filename)
        else:
            print("Generated "+filename)
