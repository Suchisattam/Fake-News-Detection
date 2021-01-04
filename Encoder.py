import torch
import torch.nn as nn

from ConvS2S import ConvEncoder
from Attention import MultiHeadAttention, PositionFeedforward

class Encoder(nn.Module): # 1 Mark
	def __init__(self, conv_layers, hidden_dim, feed_forward_dim=2048):
		super(Encoder, self).__init__()
		# Your code here
		self.conv= ConvEncoder(input_dim= hidden_dim, num_layers= conv_layers)
		self.attention= MultiHeadAttention(hid_dim= hidden_dim, n_heads= 16)
		self.feed_forward= PositionFeedforward(hid_dim= hidden_dim, feedForward_dim= feed_forward_dim)


	def forward(self, input):
		"""
		Forward Pass of the Encoder Class
		:param input: Input Tensor for the forward pass. 
		"""
		# Your code here
		out= self.conv.forward(input)
		out= self.attention.forward(out,out,out)
		out= self.feed_forward.forward(out)
		return out


