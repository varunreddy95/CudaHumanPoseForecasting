import torch.nn as nn
import torch


class SeqEncoder(nn.Module):
    def __init__(self, encoding_dim) -> None:
        super(SeqEncoder, self).__init__()
        self.linear = nn.Linear(34, encoding_dim)
        self.lstm1 = nn.LSTMCell(encoding_dim, encoding_dim)
        self.lstm2 = nn.LSTMCell(encoding_dim, encoding_dim)
        self.hidden_dim = encoding_dim


    def forward(self, x):
        device = x.device
        bsz = x.size(0)

        h_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)

        encoded_input = self.linear(x)

        for time_step in encoded_input.split(1, dim=1):
            h_t, c_t = self.lstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states


        return h_t2, c_t2


class SeqDecoder(nn.Module):
    def __init__(self, encoding_dim, zeros=True) -> None:
        super(SeqDecoder, self).__init__()
        self.linear = nn.Linear(encoding_dim, 34)
        self.lstm1 = nn.LSTMCell(encoding_dim, encoding_dim)
        self.lstm2 = nn.LSTMCell(encoding_dim, encoding_dim)
        self.encoding_dim = encoding_dim
        self.zeros = zeros

    def forward(self, h, c, future_preds=10):
        prediction_list = []

        device = h.device
        bsz = h.size(0)

        h_t = h
        c_t = c
        h_t2 = h
        c_t2 = c

        if self.zeros:

            start_token = torch.zeros(bsz, self.encoding_dim, dtype=torch.float32, device=device)

        else:
            start_token = torch.ones(bsz, self.encoding_dim, dtype=torch.float32, device=device)

        output = start_token

        for i in range(future_preds):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            output = h_t
            prediction_list.append(torch.unsqueeze(output, 1))

        prediction_tensor = torch.cat(prediction_list, dim=1)


        return self.linear(prediction_tensor)


class Seq2Seq(nn.Module):
    def __init__(self, encoding_dim, zeros=True) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = SeqEncoder(encoding_dim)
        self.decoder = SeqDecoder(encoding_dim, zeros)

    def forward(self, x):
        h_t, c_t = self.encoder(x)

        prediction = self.decoder(h_t, c_t)

        return prediction



if __name__ == "__main__":
    model = Seq2Seq(256)
    input = torch.rand((2,10,34))

    output = model(input)
