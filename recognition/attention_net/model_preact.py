import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Conv_ReLU(nn.Module):
    def __init__(self, nin, nout, ks, ss, ps, has_bn, has_bias=False):
        super(Conv_ReLU, self).__init__()
        if has_bn:
            self.subnet = nn.Sequential(
            nn.BatchNorm2d(nin),
            nn.ReLU(inplace=True),
            nn.Conv2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias),
            )
        else:
            self.subnet = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias),
            )

    def forward(self, x):
        return self.subnet(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class LSTM_Att(nn.Module):
    """docstring for LSTM_Att"""
    def __init__(self, input_size,hidden_size,attn_size,num_classes,context_vector):
        super(LSTM_Att, self).__init__()
        self.hidden_size = hidden_size #256
        self.input_size = input_size #512
        self.output_size = num_classes #37
        self.attn_size = attn_size # 256

        self.linear_s = nn.Linear(self.hidden_size , self.attn_size)
        self.linear_h = nn.Linear(self.input_size , self.attn_size)
        self.linear_v = nn.Linear(self.attn_size, 1, bias= False)
        self.tanh = nn.Tanh()

        self.rnn = nn.LSTMCell(input_size = self.input_size+num_classes, hidden_size = self.hidden_size)
        self.out1 = nn.Linear(self.hidden_size, self.output_size)
        self.out2 = nn.Linear(self.input_size+num_classes,self.output_size)

        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda:0")
        self.context_vector = context_vector
        
        f_ind= self.rnn.bias_hh.size(0)//4
        self.rnn.bias_hh.data[f_ind:2*f_ind].fill_(1.)
        self.rnn.bias_ih.data[f_ind:2*f_ind].fill_(1.)

    def forward(self,x,labels):
        batch_size = x.shape[0]
        assert len(labels)%batch_size==0, "length of labels {} is not divisible by batch_size {}".format(len(labels), batch_size)

        max_len = x.shape[1]
        char_embeddings = torch.eye(self.output_size).cuda()
        

        outputs = torch.zeros(batch_size,max_len,self.output_size).cuda()
        gts =torch.zeros(batch_size,max_len,self.input_size).cuda()
        c_t = torch.zeros(batch_size,self.hidden_size).cuda()
        s_t = torch.zeros(batch_size,self.hidden_size).cuda()
        targets = torch.zeros(batch_size, max_len+1).long().cuda()
        for b_ind in range(batch_size):
            targets[b_ind][1:] = labels[b_ind*max_len:(b_ind+1)*max_len]

        for cell in range(max_len):

            attn_Ws = torch.stack([self.linear_s(s_t)]*x.shape[1],dim=1) # [batch, dim_att] 
            attn_Vh= self.linear_h(x) # [batch,seq_len,dim_att]
            et = self.linear_v(self.tanh(torch.add(attn_Ws,attn_Vh))).squeeze(1) #[batch,seq_len,1]
            et = torch.transpose(self.softmax(et),1,2)
            gt = torch.matmul(et,x).squeeze(1) #[batch,dim_input]
            if self.context_vector: gts[:,cell] = gt
            if self.training:
                tf_embeddings = char_embeddings.index_select(0, targets.transpose(0,1)[cell]).cuda()
                gt = torch.cat([gt, tf_embeddings], 1)
            else:
                if cell == 0:pre_pred = torch.zeros(1).long().cuda()
                else: pre_pred = torch.argmax(y_t,dim =1)

                pred_embeddings= char_embeddings.index_select(0, pre_pred).cuda()
                gt = torch.cat([gt, pred_embeddings], 1)

            s_t,c_t = self.rnn(gt,(s_t,c_t))
            y_t =self.out1(s_t) +self.out2(gt) #[batch,classes]
            outputs[:,cell] = y_t  # [batch,seq,classes]


        if self.context_vector: return outputs,gts
        else: return outputs 


class AN(nn.Module):
    def __init__(self, args):
        super(AN, self).__init__()
        self.nClasses = args.nClasses
        self.context_vector = args.context_vector
        self.blstm = nn.LSTM(input_size = 512, hidden_size = 256, num_layers=2,
            dropout = 0.2, bidirectional = True, batch_first = True)
        self.lstm_att = LSTM_Att(512,256,256,args.nClasses,self.context_vector)
        self.softmax = nn.Softmax(dim=-1)

        self.conv1_x = nn.Sequential(
            Conv_ReLU(3, 32, 3, 1, 1, False), 
            Conv_ReLU(32, 64, 3, 1, 1, True)) 

        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(2,2), 
            self.ensemble_blocks(64,128,1),
            Conv_ReLU(128,128, 3, 1, 1, True))

        self.conv3_x = nn.Sequential(      
            nn.MaxPool2d(2, 2),      
            self.ensemble_blocks(128,256,2),                                  
            Conv_ReLU(256, 256, 3, 1, 1, True))

        self.conv4_x = nn.Sequential(
            nn.MaxPool2d((2,2),(2,1), (0,1)),
            self.ensemble_blocks(256,512,5),    
            Conv_ReLU(512, 512, 3, 1, 1, True))      

        self.conv5_x = nn.Sequential(
            self.ensemble_blocks(512,512,3),    
            Conv_ReLU(512, 512, (2,2), (2,1), (0,1), True),
            Conv_ReLU(512, 512, (2,2), (1,1), (0,0), True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def ensemble_blocks(self, inplanes, outplanes, blocks, stride=1):
        # planes: channels of filters
        # blocks: number of ResBlocks, each block has two conv layers 
        # self.inplanes: set to constant number(64), and downsample if input channel != output channel 

        downsample = None
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(inplanes, outplanes, stride),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(BasicBlock(inplanes, outplanes,stride, downsample))
        inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(BasicBlock(inplanes, outplanes))

        return nn.Sequential(*layers)




    def forward(self, x, labels):
        ## norm data
        x = (x/255.0 - 0.5)/0.5
        # cnn
        x = self.conv1_x(x)
        x = self.conv2_x(x) 
        x = self.conv3_x(x) 
        x = self.conv4_x(x) 
        x = self.conv5_x(x) 
        x = x.squeeze(2)
        x = torch.transpose(x,1,2)
        # rnn
        x,_ = self.blstm(x)
        if self.context_vector:
            x,gts = self.lstm_att(x,labels)
            return x,gts

        else: 
            x = self.lstm_att(x,labels)
        if not self.training:
            x = self.softmax(x)
        return x 

        
        