import torch.nn as nn
import torch
from src.configs.config import cfg
from torch.autograd import Variable

#Code modified from https://github.com/ndrplz/ConvLSTM_pytorch
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

#convlstm modified based on https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        # print(f'{input_tensor.shape=}')
        # print(f'{h_cur.shape=}')
        combined = torch.cat([input_tensor, h_cur], dim=1)
        #print(f'{combined.shape=}')
        #print(self.conv_gates)
        #print("######################################")
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

# Just in case we need multiple layer of conv.
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, encode=True):
        super(ConvBlock, self).__init__()
        self.encode = encode
        if self.encode:
            self.convlayer = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding),
                                       nn.BatchNorm2d(out_channel))
                                       #,nn.LeakyReLU(negative_slope=0.01, inplace=True))

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        else:  # for decoding
            self.convlayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channel),)
                #nn.LeakyReLU(negative_slope=0.01, inplace=True))
            self.pool = nn.MaxUnpool2d(kernel_size=2, stride=2)


    def forward(self, x):
        return self.convlayer(x)


class Encoder(nn.Module):
    def __init__(self, encode_dim, conv_sz) -> None:
        super(Encoder, self).__init__()
        self.encode_dim = encode_dim
        self.conv_sz = conv_sz       # 64, 128
        self.encode_layer1 = ConvBlock(in_channel=17, out_channel=conv_sz[0], kernel_size=3, padding=1, encode=True)
        self.encode_layer2 = ConvBlock(in_channel=conv_sz[0], out_channel=conv_sz[1], kernel_size=3, padding=1, encode=True)
        #self.encode_layer3 = ConvBlock(in_channel=conv_sz[1], out_channel=conv_sz[2], kernel_size=3, padding=1, encode=True)


    def forward(self, x):
        B,N,C,H,W = x.shape
        x = x.view(B*N, C, H, W)
        en_x = self.encode_layer1(x)
        en_x = self.encode_layer2(en_x)
        #en_x = self.encode_layer3(en_x)
        en_x = en_x.view(B, N, self.encode_dim, 64, 64)         # here after 3 max pooling 64 -> 8
        return en_x


class Decoder(nn.Module):
    def __init__(self, encode_dim, conv_sz) -> None:
        super().__init__()
        self.encode_dim = encode_dim
        self.conv_sz = conv_sz  # 256, 128, 64
        self.decode_layer = ConvBlock(in_channel=encode_dim, out_channel=conv_sz[0], kernel_size=3, padding=1, encode=False)
        self.decode_layer2 = ConvBlock(in_channel=conv_sz[0], out_channel=17, kernel_size=3, padding=1, encode=False)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        de_x = self.decode_layer(x)
        de_x = self.decode_layer2(de_x)
        de_x = de_x.view(B, N, 17, H, W)
        return de_x

class StateSpaceModel_Cell(nn.Module):
    def __init__(self, encode_dim, hidden_dim, residual=False) -> None:
        super(StateSpaceModel_Cell, self).__init__()
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(encode_dim, [64, 128])

        self.hm_sz = 64
        self.residual = residual
        self.convlstm1 = ConvLSTMCell(input_dim=self.encode_dim, hidden_dim=self.hidden_dim, kernel_size=(3,3), bias=True)
        self.convlstm2 = ConvLSTMCell(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, kernel_size=(3, 3),
                                      bias=True)

        self.decoder = Decoder(encode_dim, [128, 64])
        '''
        self.convgru1 = ConvGRUCell(input_size=(self.hm_sz, self.hm_sz), input_dim=self.encode_dim, hidden_dim=self.hidden_dim,
                                    kernel_size=(3,3), bias=True, dtype=torch.cuda.FloatTensor)
        self.convgru2 = ConvGRUCell(input_size=(self.hm_sz, self.hm_sz), input_dim=self.encode_dim, hidden_dim=self.hidden_dim,
                                    kernel_size=(3, 3), bias=True, dtype=torch.cuda.FloatTensor)
        '''

    def forward(self, x, future_pred=9):
        bsz = x.size(0)
        outputs = []
        device = x.device

        h_t = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)

        c_t = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz,dtype=torch.float32, device=device)

        encoded_input = self.encoder(x)     #BSZ frames enc_dim 64 64

        for time_step in encoded_input.split(1, dim=1):
            h_t, c_t = self.convlstm1(torch.squeeze(time_step), (h_t,c_t))  # initial hidden and cell states
            h_t2, c_t2 = self.convlstm2(h_t, (h_t2,c_t2))  # new hidden and cell states
            output = h_t2

        outputs.append(torch.unsqueeze(output, 1))
        for i in range(future_pred):
            h_t, c_t = self.convlstm1(output, (h_t,c_t))
            h_t2, c_t2 = self.convlstm2(h_t, (h_t2,c_t2))
            output = h_t2
            outputs.append(torch.unsqueeze(output, 1))

        prediction_tensor = torch.cat(outputs, 1)

        decoded_prediction = self.decoder(prediction_tensor)

        if self.residual:

            de_clone = decoded_prediction.clone()

            de_clone[:, 0, ...] =  decoded_prediction[:, 0, ...] + x[:, 9, ...]

            for i in range(9):                # IndexError: index 9 is out of bounds for dimension 1 with size 9 when using range 9
                 de_clone[:, i + 1, ...] = de_clone[:, i + 1, ...] + de_clone[:, i, ...]

            return de_clone

        else:
            return decoded_prediction

class Encoder_AG(nn.Module):
    def __init__(self, encode_dim, conv_sz) -> None:
        super().__init__()
        self.encode_dim = encode_dim
        self.conv_sz = conv_sz  # 64, 128
        self.encode_layer1 = ConvBlock(in_channel=17, out_channel=conv_sz[0], kernel_size=3, padding=1, encode=True)
        self.encode_layer2 = ConvBlock(in_channel=conv_sz[0], out_channel=conv_sz[1], kernel_size=3, padding=1,
                                       encode=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H, W)
        en_x = self.encode_layer1(x)
        en_x = self.encode_layer2(en_x)
        # en_x = self.encode_layer3(en_x)
        en_x = en_x.view(B, self.encode_dim, 64, 64)  # here after 3 max pooling 64 -> 8
        return en_x

class Decoder_AG(nn.Module):
    def __init__(self, encode_dim, conv_sz) -> None:
        super().__init__()
        self.encode_dim = encode_dim
        self.conv_sz = conv_sz  # 256, 128, 64
        self.decode_layer = ConvBlock(in_channel=encode_dim, out_channel=conv_sz[0], kernel_size=3, padding=1, encode=False)
        self.decode_layer2 = ConvBlock(in_channel=conv_sz[0], out_channel=17, kernel_size=3, padding=1, encode=False)

    def forward(self, x):
        B, C, H, W = x.shape
        #x = x.view(B * N, C, H, W)
        de_x = self.decode_layer(x)
        de_x = self.decode_layer2(de_x)
        de_x = de_x.view(B, 17, H, W)
        return de_x

class AutoregressiveModel_cell(nn.Module):
    def __init__(self, encode_dim, hidden_dim) -> None:
        super(AutoregressiveModel_cell, self).__init__()
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(encode_dim, [64, 128])
        self.encoder_mid = Encoder_AG(encode_dim, [64, 128])
        self.convlstm1 = ConvLSTMCell(input_dim=self.encode_dim, hidden_dim=self.hidden_dim, kernel_size=(3, 3),
                                      bias=True)
        self.convlstm2 = ConvLSTMCell(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, kernel_size=(3, 3),
                                      bias=True)
        self.decoder = Decoder_AG(encode_dim, [128, 64])
        #self.decoder = Decoder(encode_dim, [128, 64])
        self.hm_sz = 64
        '''
        self.convgru1 = ConvGRUCell(input_size=(self.hm_sz, self.hm_sz), input_dim=self.encode_dim,
                                    hidden_dim=self.hidden_dim,
                                    kernel_size=(3, 3), bias=True, dtype=torch.cuda.FloatTensor)
        self.convgru2 = ConvGRUCell(input_size=(self.hm_sz, self.hm_sz), input_dim=self.encode_dim,
                                    hidden_dim=self.hidden_dim,
                                    kernel_size=(3, 3), bias=True,
                                    dtype=torch.cuda.FloatTensor)  # torch.cuda.FloatTensor
        '''

    def forward(self, x, future_pred=9):
        bsz = x.size(0)
        prediction_list = []
        device = x.device
        h_t = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)

        c_t = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)
        encoded_input = self.encoder(x)

        for time_step in encoded_input.split(1, dim=1):
            h_t, c_t = self.convlstm1(torch.squeeze(time_step), (h_t,c_t))  # initial hidden and cell states
            h_t2, c_t2 = self.convlstm2(h_t, (h_t2,c_t2))  # new hidden and cell states
            output = h_t2              # BSZ hidden 64 64

        #output = torch.unsqueeze(output, dim=1)
        de_output = self.decoder(output)

        prediction_list.append(torch.unsqueeze(de_output, 1))

        for i in range(future_pred):
            output = self.encoder_mid(de_output)
            h_t, c_t = self.convlstm1(torch.squeeze(output), (h_t,c_t))  # initial hidden and cell states
            h_t2,c_t2 = self.convlstm2(h_t, (h_t2,c_t2))  # new hidden and cell states
            output = h_t2

            de_output = self.decoder(output)


            prediction_list.append(torch.unsqueeze(de_output, dim=1))

        prediction_tensor = torch.cat(prediction_list, dim=1)


        return prediction_tensor


if __name__ == '__main__':
    #print(torch.__version__)
    #model = AutoregressiveModel_cell(encode_dim=128, hidden_dim=128).to(cfg.DEVICE)
    model2 = StateSpaceModel_Cell(encode_dim=128, hidden_dim=128).to(cfg.DEVICE)
    test = torch.randn(2, 10, 17, 64, 64).to(cfg.DEVICE)
    # #output = model(test)
    output2 = model2(test)
    # #print(f'{output.shape=}')
    print(f'{output2.shape=}')



'''

class Model4Heatmap(nn.Module):
    def __init__(self, encode_dim, hidden_dim) -> None:
        super(Model4Heatmap, self).__init__()
        self.device = cfg.DEVICE
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim

        # whole model related structure
        self.encoder = Encoder(self.encode_dim)
        self.decoder = Decoder(self.encode_dim)
        self.shortcut = nn.Identity()
        self.activation = nn.ReLU()

        # lstm relevant parameters
        self.num_layers = 1
        self.bsz = cfg.BATCH_SIZE
        self.init_mode = cfg.HC_INIT_MODE
        #self.GRU_layer = nn.GRU((64,64), self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        #self.lstm_layer = ConvLSTM(input_dim=self.encode_dim, hidden_dim=self.hidden_dim,
        #                           kernel_size=(3,3), num_layers=1, batch_first=True)


    def forward(self, x, future_pred=10):
        # Input Size: BSZ, num_frames, n_channels=17, H, W
        # print("Input shape: {}".format(x.shape))

        en_x = self.encoder(x)      # BSZ  frames encode_dim h w
        #print("After Encoder shape should be [32, 10, 64, 64, 64] and we have {}".format(en_x.shape))
        output, hidden_states = self.lstm_layer(en_x)
        encoded_input = output[0].clone()
        h_states = hidden_states.copy()
        #print(f"{encoded_input.shape=}")

        prediction_list = []
        output = torch.unsqueeze(encoded_input[:, 9, ...], dim=1)
        #print(f"{output.shape=}")
        for i in range(future_pred):
            #print("Iteration {}".format(i))
            output_li, hidden_states = self.lstm_layer(output, h_states)
            output = output_li[0].clone()
            h_states = hidden_states.copy()
            prediction_list.append(output)

        prediction_tensor = torch.cat(prediction_list, dim=1)

        decoded_prediction = self.decoder(prediction_tensor)
        output = decoded_prediction.clone()

        output[:, 0, ...] = self.activation(decoded_prediction[:, 0, ...] + x[:, 9, ...])
        #decoded_prediction[:, 0, ...] = decoded_prediction[:, 0, ...] + x[:, 9, ...]

        for i in range(9):
            output[:, i + 1, ...] = self.activation(decoded_prediction[:, i + 1, ...] + decoded_prediction[:, i, ...])

        return output

'''