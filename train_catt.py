import torch
import pickle
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from tashkeel_dataset import TashkeelDataset, PrePaddingDataLoader
from tashkeel_tokenizer import TashkeelTokenizer

from peft import LoraConfig, get_peft_model, TaskType

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

# Model's Configs
model_type = 'ed' # 'eo' for Encoder-Only OR 'ed' for Encoder-Decoder
dl_num_workers = 2
batch_size = 16
max_seq_len = 512
threshold = 0.6

LORAN_R = 8 # Rank (dimensionality) of the two low-rank matrices (A and B) that replace the full weight matrix update.
## lower value means less accuracy, common values: 4,8,16,32
LORAN_ALPHA = 16 #scaling factor to normalize the output of the LoRA updates
# commonlyu set to twice the rank.
LORAN_DROPOUT = 0.05 # to avoid overfitting

# Pretrained Char-Based BERT
pretrained_mlm_pt = "models/char_bert_model_pretrained.pt" # Use None if you want to initialize weights randomly OR the path to the char-based BERT
#pretrained_mlm_pt = 'char_bert_model_pretrained.pt'

MODEL_OUTPUT_DIR = f'catt_{model_type}_model_v1_lora/' # <-- MODIFIED

train_txt_folder_path = 'data/train/'
val_txt_folder_path = 'data/val/'
# test_txt_folder_path = 'dataset/test'


if model_type == 'ed':
    from ed_pl import TashkeelModel
else:
    from eo_pl import TashkeelModel


tokenizer = TashkeelTokenizer()

print('Creating Train Dataset...')
train_dataset = TashkeelDataset(train_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
print('Creating Train Dataloader...')
train_dataloader = PrePaddingDataLoader(tokenizer, train_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=True)

print('Creating Validation Dataset...')
val_dataset = TashkeelDataset(val_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
print('Creating Validation Dataloader...')
val_dataloader = PrePaddingDataLoader(tokenizer, val_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=False)

# print('Creating Test Dataset...')
# test_dataset = TashkeelDataset(test_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
# print('Creating Test Dataloader...')
# test_dataloader = PrePaddingDataLoader(tokenizer, test_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=False)

print('Creating Model...')
model = TashkeelModel(tokenizer, max_seq_len=max_seq_len, n_layers=6, learnable_pos_emb=False)

# Use the pretrained weights of the char-based BERT model to initialize the model
if not pretrained_mlm_pt is None:
    missing = model.transformer.load_state_dict(torch.load(pretrained_mlm_pt), strict=False)
    print(f'Missing layers: {missing}')

# --- START LoRA INTEGRATION (NEW CODE BLOCK) ---
print('Applying LoRA to the Transformer model...')
lora_config = LoraConfig( ## LoraCondfig holds the hyperparameters for loRA adapters
    r=LORAN_R,
    lora_alpha=LORAN_ALPHA,
    target_modules=[
        "w_q", "w_k", "w_v", "w_concat",
        "linear1", "linear2"
    ], # Target the key Self-Attention layers
    lora_dropout=LORAN_DROPOUT,
    bias="none", # the bias terms will not be trained or affected by the LoRA update
    #task_type=TaskType.CAUSAL_LM, # Sequence-to-Sequence for Encoder-Decoder
)

# 2. Wrap the base model: freezes the 72.3MB BERT weights and adds/unfreezes the small LoRA adapters
model.transformer = get_peft_model(model.transformer, lora_config)

# 3. Print verification: this confirms that the trainable parameters are now very small
model.transformer.print_trainable_parameters()
# --- END LoRA INTEGRATION ---

# This is to freeze the encoder weights
#freeze(model.transformer.encoder)

#dirpath = f'catt_{model_type}_model_v1/'

checkpoint_callback = ModelCheckpoint(dirpath=MODEL_OUTPUT_DIR, save_top_k=10, save_last=True,
                                      monitor='val_der',
                                      filename=f'catt_{model_type}_model' + '-{epoch:02d}-{val_loss:.5f}-{val_der:.5f}')

print('Creating Trainer...')

logs_path = f'{MODEL_OUTPUT_DIR}/logs'

print('#'*100)
print(model)
print('#'*100)

early_stop = EarlyStopping(
    monitor="val_der",
    mode="min",      # lower is better
    patience=6,      # stop if no improvement for 6 epochs
    verbose=True
)

trainer = Trainer(
    #accelerator="cpu",
    accelerator="cuda",
    devices=-1,
    max_epochs=32,
    callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, early_stop],
    precision=16,
    logger=CSVLogger(save_dir=logs_path),
#    strategy="ddp_find_unused_parameters_false"
    )

#ckpt_path = 'YOUR_CKPT_PATH_GOES_HERE'
#trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
ckpt_path = "/kaggle/working/catt/catt/catt_ed_model_v1_lora/last.ckpt"
trainer.fit(
    model,
    train_dataloader,
    val_dataloader,
    ckpt_path=ckpt_path
)