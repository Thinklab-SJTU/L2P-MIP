import torch
print(torch.__version__)
import torch.nn.functional as F
import torch_geometric
import numpy as np

# emb_size = 32

class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[
            -1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size):
        super().__init__('add')
        # emb_size = 128

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output


class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GCNPolicy(BaseModel):
    def __init__(self, mode, no_grad, single_layer, emb_size):
        super().__init__()
        self.mode = mode
        self.no_grad = no_grad
        self.single_layer = single_layer
        self.emb_size = emb_size
        # emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        self.round_class = 4
        self.time_class = 4

        self.output_size = 14 + 14 * self.round_class + 14 * self.time_class

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        # self.fc1_single = torch.nn.Sequential(
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(emb_size, 14)
        # )
        #
        # self.fc2_single = torch.nn.Sequential(
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(emb_size + 14, 56)
        # )
        #
        # self.fc3_single = torch.nn.Sequential(
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(emb_size + 14, 56)
        # )
        if self.single_layer:
            self.fc1 = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size + 14, 14)
            )

            self.fc2 = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size + 14, 56)
            )

            self.fc3 = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size + 14, 56)
            )

            self.fc1_1 = torch.nn.Sequential(
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.Tanh(),
            )

            self.fc1_2 = torch.nn.Linear(emb_size, 14)

            self.fc2_ = torch.nn.Sequential(
                torch.nn.Linear(emb_size * 2, 56)
            )

            self.fc3_ = torch.nn.Sequential(
                torch.nn.Linear(emb_size * 2, 56)
            )
        else:
            self.fc1 = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size, 14)
            )

            self.fc2 = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size + 14, emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size, 56)
            )

            self.fc3 = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size + 14, emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size, 56)
            )

            self.fc1_1 = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.Tanh(),
            )

            self.fc1_2 = torch.nn.Linear(emb_size, 14)

            self.fc2_ = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size * 2, emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size, 56)
            )

            self.fc3_ = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size * 2, emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(emb_size, 56)
            )

    def forward(self, inputs):
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.
        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.
        Inputs
        ------
        constraint_features: 2D float tensor
            Constraint node features (n_constraints x n_constraint_features)
        edge_indices: 2D int tensor
            Edge constraint and variable indices (2, n_edges)
        edge_features: 2D float tensor
            Edge features (n_edges, n_edge_features)
        variable_features: 2D float tensor
            Variable node features (n_variables, n_variable_features)
        n_cons_per_sample: 1D int tensor
            Number of constraints for each of the samples stacked in the batch.
        n_vars_per_sample: 1D int tensor
            Number of variables for each of the samples stacked in the batch.
        """
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features,
                                               constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # output = self.output_module(variable_features).squeeze(-1)

        count = 0
        pred = torch.zeros([len(n_vars_per_sample), self.emb_size], device=variable_features.device)
        for i in range(len(n_vars_per_sample)):
            pred[i, :] = torch.mean(variable_features[count: count + n_vars_per_sample[i].cpu().detach().numpy()][:],
                                    dim=0)
            count += n_vars_per_sample[i].cpu().detach().numpy()

        if self.mode == 0:
            real_output1_ = self.fc1(pred)
            if self.no_grad:
                real_output1 = real_output1_.detach()
            else:
                real_output1 = real_output1_
            pred_ = torch.concat((pred, real_output1), dim=1)
            real_output2 = self.fc2(pred_)
            real_output3 = self.fc3(pred_)
        else:
            real_output1_hidden = self.fc1_1(pred)
            real_output1 = self.fc1_2(real_output1_hidden)
            if self.no_grad:
                real_output1_ = real_output1_hidden.detach()
            else:
                real_output1_ = real_output1_hidden
            pred_ = torch.concat((pred, real_output1_), dim=1)
            real_output2 = self.fc2_(pred_)
            real_output3 = self.fc3_(pred_)
        real_output = torch.concat((real_output1, real_output2, real_output3), dim=1)
        return real_output
