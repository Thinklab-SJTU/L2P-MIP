import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMPolicy(nn.Module):
    """LSTM controller, inspired by AutoAugment."""

    def __init__(self, configs):
        super(LSTMPolicy, self).__init__()
        self.configs = configs
        self.lstm = nn.LSTM(
            1, configs["lstm_units"], num_layers=2
        )  # (BATCHES, TIME_STEPS, lstm_units)
        self.op_type, self.op_prob = [], []

        self.line = nn.Linear(100, 1)

        for i in range(configs["sub_policies"]):
            self.op_type.append(
                nn.Linear(configs["lstm_units"], configs["sub_policy_op_priority"])
            )
            self.op_prob.append(
                nn.Linear(configs["lstm_units"], configs["sub_policy_op_round"])
            )

    def to_cuda(self):
        self.cuda()
        for i in range(self.configs["sub_policies"]):
            self.op_type[i].cuda()
            self.op_prob[i].cuda()

    def forward(self, A):

        # do LSTM layer
        
        x = self.line(A)
        x = x.unsqueeze(1)

        lstm_outputs, (_, _) = self.lstm(x)

        # lstm_outputs is of shape (sub_policies, batches, lstm_units)
        assert lstm_outputs.shape == (
            A.shape[0],
            1,
            self.configs["lstm_units"],
        )

        # do fully connected layer
        xx = lstm_outputs[-1]

        op_type_outputs = torch.zeros((self.configs["sub_policies"], 1, self.configs["sub_policy_ops"], self.configs["sub_policy_op_priority"]), dtype=torch.float32).cuda()
        op_prob_outputs = torch.zeros((self.configs["sub_policies"], 1, self.configs["sub_policy_ops"], self.configs["sub_policy_op_round"]), dtype=torch.float32).cuda()

        for i in range(self.configs["sub_policies"]):
            op_type_outputs[i][0] = self.op_type[i](xx)
            op_prob_outputs[i][0] = self.op_prob[i](xx)


        #op_type_outputs = torch.tensor(op_type_outputs).unsqueeze(2)
        #op_prob_outputs = torch.stack(op_prob_outputs, dim=2)


        # each of the above is of shape (sub_policies, batches, sub_policy_ops, XXX)
        assert op_type_outputs.shape == (
            self.configs["sub_policies"],
            1,
            self.configs["sub_policy_ops"],
            self.configs["sub_policy_op_priority"],
        )

        return op_type_outputs, op_prob_outputs


class LSTMValue(nn.Module):
    def __init__(self, configs):
        super(LSTMValue, self).__init__()
        self.configs = configs
        self.lstm = nn.LSTM(
            1, configs["lstm_units"]
        )  # (BATCHES, TIME_STEPS, lstm_units)
        self.fc = nn.Linear(configs["lstm_units"], 1)

    def to_cuda(self):
        self.cuda()
        for i in range(self.configs["sub_policy_ops"]):
            self.op_type[i].cuda()
            self.op_prob[i].cuda()
            self.op_mag1[i].cuda()
            self.op_mag2[i].cuda()

    def forward(self, zero):

        # x is a zero tensor of shape (sub_policies, batches, FEATURES=1)
        assert torch.min(zero) == 0 and torch.max(zero) == 0
        assert zero.shape == (self.configs["sub_policies"], 1, 1)

        # do LSTM layer
        _, (h, _) = self.lstm(zero)

        # do fully-connected layer
        output = self.fc(h)

        assert output.shape == (1, 1, 1)

        output = output.view(-1)
        return output
