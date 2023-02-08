from src.models.seq2seq import Seq2Seq
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super(Discriminator, self).__init__()

        self.encode_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(34, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)
        self.lstm1 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        bsz = x.size(0)
        outputs = []

        device = x.device

        h_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)

        encoded_input = self.encoder(x)

        for time_step in encoded_input.split(1, dim=1):
            h_t, c_t = self.lstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states

        output = h_t2

        #return self.relu(self.decoder(output))
        return self.decoder(output)

if __name__ == "__main__":
    input = torch.rand((2, 10, 34))
    model = Discriminator(256)

    output = model(input)
    print(output.size())
    print(output)