import torch.nn as nn
import torch

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

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, encode=True):
        super(ConvBlock, self).__init__()
        self.encode = encode
        if self.encode:
            self.convlayer = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding),
                                       nn.BatchNorm2d(out_channel),)
                                       #nn.LeakyReLU(negative_slope=0.01, inplace=True))

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
        self.conv_sz = conv_sz  # 128, 64
        self.decode_layer = ConvBlock(in_channel=encode_dim, out_channel=conv_sz[0], kernel_size=3, padding=1, encode=False)
        self.decode_layer2 = ConvBlock(in_channel=conv_sz[0], out_channel=17, kernel_size=3, padding=1, encode=False)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        de_x = self.decode_layer(x)
        de_x = self.decode_layer2(de_x)
        de_x = de_x.view(B, N, 17, H, W)
        return de_x


class SeqEncoder(nn.Module):
    def __init__(self, encoding_dim) -> None:
        super(SeqEncoder, self).__init__()
        self.encoder = Encoder(encoding_dim, [64, 128])
        self.convlstm1 = ConvLSTMCell(input_dim=encoding_dim, hidden_dim=encoding_dim, kernel_size=(3, 3),
                                      bias=True)
        self.convlstm2 = ConvLSTMCell(input_dim=encoding_dim, hidden_dim=encoding_dim, kernel_size=(3, 3),
                                      bias=True)
        self.hidden_dim = encoding_dim
        self.hm_sz = 64


    def forward(self, x):
        device = x.device
        bsz = x.size(0)

        h_t = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)

        c_t = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(bsz, self.hidden_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)

        encoded_input = self.encoder(x)

        for time_step in encoded_input.split(1, dim=1):
            h_t, c_t = self.convlstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.convlstm2(h_t, (h_t2, c_t2)) # new hidden and cell states

        return h_t2, c_t2


class SeqDecoder(nn.Module):
    def __init__(self, encoding_dim, zeros=True) -> None:
        super(SeqDecoder, self).__init__()
        self.decoder = Decoder(encoding_dim, [128, 64])
        self.convlstm1 = ConvLSTMCell(input_dim=encoding_dim, hidden_dim=encoding_dim, kernel_size=(3, 3),
                                      bias=True)
        self.convlstm2 = ConvLSTMCell(input_dim=encoding_dim, hidden_dim=encoding_dim, kernel_size=(3, 3),
                                      bias=True)
        self.encoding_dim = encoding_dim
        self.zeros = zeros
        self.hm_sz = 64

    def forward(self, h, c, future_preds=10):
        prediction_list = []

        device = h.device
        bsz = h.size(0)

        h_t = h
        c_t = c
        h_t2 = h
        c_t2 = c

        if self.zeros:

            start_token = torch.zeros(bsz,self.encoding_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)

        else:
            start_token = torch.ones(bsz, self.encoding_dim, self.hm_sz, self.hm_sz, dtype=torch.float32, device=device)

        output = start_token

        for i in range(future_preds):
            h_t, c_t = self.convlstm1(output, (h_t, c_t))
            output = h_t
            prediction_list.append(torch.unsqueeze(output, 1))

        prediction_tensor = torch.cat(prediction_list, dim=1)


        return self.decoder(prediction_tensor)


class Seq2SeqHM(nn.Module):
    def __init__(self, encoding_dim, zeros=True) -> None:
        super(Seq2SeqHM, self).__init__()
        self.encoder = SeqEncoder(encoding_dim)
        self.decoder = SeqDecoder(encoding_dim, zeros)

    def forward(self, x):
        h_t, c_t = self.encoder(x)

        prediction = self.decoder(h_t, c_t)

        return prediction

if __name__ == '__main__':
    model = Seq2SeqHM(encoding_dim=128).to("cuda")
    test = torch.randn(2, 10, 17, 64, 64).to("cuda")
    out = model(test)
    assert test.shape == out.shape