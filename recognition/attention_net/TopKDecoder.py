import torch
import torch.nn.functional as F
from torch.autograd import Variable


class TopKDecoder(torch.nn.Module):
	"""
	Args:
		decoder_rnn: An object of DecoderRNN used for decoding 
		k (int): size of the beam 

	Inputs:inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
		-- **inputs** (batch,seq_len,input_size)
		-- **encoder_hidden** (batch_size,num_layers * num_directions,  hidden_size)
		-- **encoder_outputs** (batch, seq_len, hidden_size)
		-- **function**: a function used to generate symbol from hidden state
		-- **teacher_forcing_ratio**: The prob that teacher forcing will be used. 
			default = 0 

	Outpurs: decoder_outputs, decoder_hidden, ret_dict 
		-- **decoder_outputs** (batch): batch-length list of tensors with size 
			(max_length,hideen_size) containing the outputs of the decoder 
		-- **decoder_hidden** (num_layers * num_directions, batch, hidden_size)
		-- **ret_dict**: dictionary containing addition info 
	"""

	def __init__(self,decoder_rnn,k):
		super(TopKDecoder,self).__init__()
		self.rnn = decoder_rnn
		self.k = k
		self.hideen_size = self.rnn.hideen_size
		self.V = self.rnn.output_size 
		self.SOS =
		self.EOS = 

	def forward(self, inputs = None, encoder_hidden = None, encoder_outputs=None, function = F.logsoftmaxm, teacher_forcing_ratio = 0, retain_output_probs = True):
		inputs,batch_size,max_length = self.rnn._validate_args(inputs,encoder_hidden,encoder_outputs,function, teacher_forcing_ratio)



 class LSTM_att():
 	def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
 		# inference batch size 
 		batch = encoder_hidden.size(9)
