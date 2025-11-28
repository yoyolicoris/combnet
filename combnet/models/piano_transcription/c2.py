import torch
import combnet
from combnet.modules import C2, C2V2


class C2Classifier(torch.nn.Module):
    def __init__(self, n_filters=12, comb_kwargs={}):
        super().__init__()

        window_size = combnet.WINDOW_SIZE
        stride = combnet.HOPSIZE

        comb = C2(1, n_filters, sr=combnet.SAMPLE_RATE, **comb_kwargs)

        n_classes = 12
        self.layers = torch.nn.Sequential(
            comb,
            torch.nn.MaxPool1d(kernel_size=window_size, stride=stride),
            torch.nn.ELU(),
            torch.nn.Conv1d(n_filters, n_filters, 1),
            torch.nn.ELU(),
            torch.nn.Conv1d(n_filters, n_classes, 1),
        )

    def parameter_groups(self):
        groups = {}
        groups["f0"] = [self.layers[0].parametrizations.f.original]
        groups["main"] = list(self.layers[1:].parameters())
        return groups

    def forward(self, audio):
        audio = torch.nn.functional.pad(
            audio, (combnet.WINDOW_SIZE // 2, combnet.WINDOW_SIZE // 2)
        )
        return self.layers(audio)


class C2V2Classifier(torch.nn.Module):
    def __init__(self, n_filters=12, comb_kwargs={}):
        super().__init__()

        window_size = combnet.WINDOW_SIZE
        stride = combnet.HOPSIZE

        comb = C2V2(1, n_filters, sr=combnet.SAMPLE_RATE, **comb_kwargs)

        n_classes = 12
        self.layers = torch.nn.Sequential(
            comb,
            torch.nn.MaxPool1d(kernel_size=window_size, stride=stride),
            torch.nn.ELU(),
            torch.nn.Conv1d(n_filters, n_filters, 1),
            torch.nn.ELU(),
            torch.nn.Conv1d(n_filters, n_classes, 1),
        )

    def parameter_groups(self):
        groups = {}
        groups["f0"] = [self.layers[0].parametrizations.f.original]
        groups["main"] = list(self.layers[1:].parameters())
        return groups

    def forward(self, audio):
        audio = torch.nn.functional.pad(
            audio, (combnet.WINDOW_SIZE // 2, combnet.WINDOW_SIZE // 2)
        )
        return self.layers(audio)
