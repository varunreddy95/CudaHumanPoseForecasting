import torch.nn as nn
import torch
from src.configs.config import cfg



class Encoder(nn.Module):
    def __init__(self, encode_dim) -> None:
        super(Encoder, self).__init__()
        self.encode_layer = nn.Linear(34, encode_dim)

    def forward(self, x):
        x = self.encode_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encode_dim) -> None:
        super().__init__()
        self.decode_layer = nn.Linear(encode_dim, 34)

    def forward(self, x):
        x = self.decode_layer(x)
        return x
        
class StateSpaceModel_Cell(nn.Module):
    def __init__(self, encode_dim, hidden_dim, teacher_forcing=False, residual_connection=True) -> None:
        super(StateSpaceModel_Cell, self).__init__()
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(encode_dim)
        self.decoder = Decoder(hidden_dim)
        self.lstm1 = nn.LSTMCell(encode_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.teacher_forcing = teacher_forcing
        self.residual_connection = residual_connection

    def forward(self, x, future_pred=9):
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

        
        outputs.append(torch.unsqueeze(output, 1))
        for i in range(future_pred):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = h_t2
            outputs.append(torch.unsqueeze(output, 1))


        prediction_tensor = torch.cat(outputs, 1)

        decoded_prediction = self.decoder(prediction_tensor)

        if self.residual_connection:

            de_clone = decoded_prediction.clone()

            de_clone[..., 0, :] = decoded_prediction[...,
                                                            0, :] + x[..., 9, :]

            for i in range(9):
                de_clone[..., i+1, :] = de_clone[...,
                                                                    i+1, :] + de_clone[..., i, :]

        # if self.residual_connection:

        #     de_clone = decoded_prediction.clone()

        #     decoded_prediction[..., 0, :] = decoded_prediction[...,
        #                                                     0, :] + x[..., 9, :]

        #     for i in range(9):
        #         decoded_prediction[..., i+1, :] = decoded_prediction[...,
        #                                                             i+1, :] + decoded_prediction[..., i, :]

            return de_clone

        return decoded_prediction

class AutoregressiveModel_Cell(nn.Module):
    def __init__(self, encode_dim, hidden_dim, teacher_forcing=False, residual_connection=True, use_relu=False) -> None:
        super(AutoregressiveModel_Cell, self).__init__()
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(encode_dim)
        self.decoder = Decoder(encode_dim)
        self.lstm1 = nn.LSTMCell(encode_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.teacher_forcing = teacher_forcing
        self.residual_connection = residual_connection
        self.relu = nn.ReLU()
        self.use_relu = use_relu

    def forward(self, x, future_pred=9, ground_truth=None):
        if not self.training or not self.teacher_forcing:
            device = x.device
            bsz = x.size(0)
            outputs = []

            #create tensors on same device as the input
            h_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
            c_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
            h_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
            c_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
            encoded_input = self.encoder(x)

            for time_step in encoded_input.split(1, dim=1):
                h_t, c_t = self.lstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states

                output = h_t2

            output = self.decoder(output)

            prediction_list = []
            #output = self.decoder(torch.unsqueeze(encoded_input[..., 9, :], dim=1))
            prediction_list.append(torch.unsqueeze(output, dim=1))

            for i in range(future_pred):
                output = self.encoder(output)
                h_t, c_t = self.lstm1(output, (h_t, c_t)) # initial hidden and cell states
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
                output = self.decoder(h_t2)

                prediction_list.append(torch.unsqueeze(output, dim=1))

            prediction_tensor = torch.cat(prediction_list, dim=1)

            if self.residual_connection:
                prediction_tensor[..., 0, :] = prediction_tensor[...,
                                                            0, :] + x[..., 9, :]

                for i in range(9):
                    prediction_tensor[..., i+1, :] = prediction_tensor[...,
                                                                        i+1, :] + prediction_tensor[..., i, :]

            if self.use_relu:
                return self.relu(prediction_tensor)
            return prediction_tensor
        
        device = x.device
        bsz = x.size(0)
        outputs = []

        #create tensors on same device as the input
        h_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        encoded_input = self.encoder(x)

        for time_step in encoded_input.split(1, dim=1):
            h_t, c_t = self.lstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states

            output = h_t2

        output = self.decoder(output)

        prediction_list = []
        #output = self.decoder(torch.unsqueeze(encoded_input[..., 9, :], dim=1))
        prediction_list.append(torch.unsqueeze(output, dim=1))

        for i in range(future_pred):
            output = self.encoder(ground_truth[...,i,:])
            h_t, c_t = self.lstm1(output, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.decoder(h_t2)

            prediction_list.append(torch.unsqueeze(output, dim=1))

        prediction_tensor = torch.cat(prediction_list, dim=1)

        if self.residual_connection:
            prediction_tensor[..., 0, :] = prediction_tensor[...,
                                                        0, :] + x[..., 9, :]

            for i in range(9):
                prediction_tensor[..., i+1, :] = prediction_tensor[...,
                                                                    i+1, :] + prediction_tensor[..., i, :]


        if self.use_relu:
            return self.relu(prediction_tensor)
        return prediction_tensor


class AutoregressiveModel_Cell_4LSTM(nn.Module):
    def __init__(self, encode_dim, hidden_dim, teacher_forcing=False, residual_connection=True) -> None:
        super(AutoregressiveModel_Cell_4LSTM, self).__init__()
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(encode_dim)
        self.decoder = Decoder(encode_dim)
        self.lstm1 = nn.LSTMCell(encode_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm3 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm4 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.teacher_forcing = teacher_forcing
        self.residual_connection = residual_connection

    def forward(self, x, future_pred=9, ground_truth=None):

        device = x.device
        bsz = x.size(0)
        outputs = []

        #create tensors on same device as the input
        h_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        h_t3 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t3 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        h_t4 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        c_t4 = torch.zeros(bsz, self.hidden_dim, dtype=torch.float32, device=device)
        encoded_input = self.encoder(x)

        for time_step in encoded_input.split(1, dim=1):
            h_t, c_t = self.lstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm3(h_t3, (h_t4, c_t4))

            output = h_t4

        output = self.decoder(output)

        prediction_list = []
        #output = self.decoder(torch.unsqueeze(encoded_input[..., 9, :], dim=1))
        prediction_list.append(torch.unsqueeze(output, dim=1))

        for i in range(future_pred):
            output = self.encoder(output)
            h_t, c_t = self.lstm1(output, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm3(h_t3, (h_t4, c_t4))
            
            output = self.decoder(h_t4)

            prediction_list.append(torch.unsqueeze(output, dim=1))

        prediction_tensor = torch.cat(prediction_list, dim=1)

        if self.residual_connection:
            prediction_tensor[..., 0, :] = prediction_tensor[...,
                                                           0, :] + x[..., 9, :]

            for i in range(9):
                prediction_tensor[..., i+1, :] = prediction_tensor[...,
                                                                    i+1, :] + prediction_tensor[..., i, :]


        return prediction_tensor

if __name__ == '__main__':
    from src.data.datasets import Hpose
    from torch.utils.data import DataLoader
    from paths import DATASET_DIR

    validation_dataset = Hpose(subset="valid")

    dataloader = DataLoader(validation_dataset, batch_size=2)

    model = AutoregressiveModel_Cell_4LSTM(256, 256).to("cuda:0")
    model.eval()

    input, gt = next(iter(dataloader))

    input, gt = input.to("cuda:0"), gt.to("cuda:0")

    output = model(input)

    print(output.size())
