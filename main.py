# 0.75 Marks. 
# To test your trainer and  arePantsonFire class, Just create random tensor and see if everything is working or not.  
from torch.utils.data import DataLoader
from utils import *
from datasets import *
from ConvS2S import *
from Attention import *
from Encoder import *
from trainer import *
from LiarLiar import *

# Your code goes here.
liar_dataset_train= dataset(prep_Data_from='train')
liar_dataset_val= dataset(prep_Data_from='val')
dataloader_train= DataLoader(dataset=liar_dataset_train, batch_size=1)
dataloader_val= DataLoader(dataset=liar_dataset_val, batch_size=1)
statement_encoder= Encoder(conv_layers=5, hidden_dim=512)
justification_encoder= Encoder(conv_layers=5, hidden_dim=512)
multiheadAttention= MultiHeadAttention(hid_dim=512,n_heads=32)
positionFeedForward= PositionFeedforward(hid_dim=512,feedForward_dim=2048)
sent_length,just_max= liar_dataset_train.get_max_lenghts()
model= arePantsonFire(sentence_encoder=statement_encoder, explanation_encoder=justification_encoder, multihead_Attention=multiheadAttention, 
					position_Feedforward=positionFeedForward, hidden_dim=512, max_length_sentence=sent_length, max_length_justification=just_max, 
					input_dim=200)
trainer(model=model, train_dataloader=dataloader_train, val_dataloader=dataloader_val, num_epochs=101, path_to_save='C:/Users/suchi/Downloads/2017B2A70585P/checkpoint',
		checkpoint_path='C:/Users/suchi/Downloads/2017B2A70585P/checkpoint',checkpoint=10, train_batch=1, test_batch=1)
# Do not change module_list , otherwise no marks will be awarded
module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val


liar_dataset_test = dataset(prep_Data_from='test')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)
