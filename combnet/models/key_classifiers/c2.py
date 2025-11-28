import torch
import combnet

from combnet.madmom import LogarithmicFilterbank
from combnet.modules import C2


class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class C2Classifier(torch.nn.Module):
    def __init__(
        self,
        n_filters=12,
        n_conv_layers=5,
        linear_channels=None,
        window_size=None,
        stride=None,
        comb_kwargs={},
    ):
        super().__init__()
        centers = None
        import numpy as np

        centers = torch.from_numpy(
            LogarithmicFilterbank(
                np.linspace(0, combnet.SAMPLE_RATE // 2, combnet.N_FFT // 2 + 1),
                num_bands=24,
                fmin=65,
                fmax=2100,
                unique_filters=True,
            ).center_frequencies
        ).float()

        if window_size is None:
            window_size = combnet.WINDOW_SIZE
        if stride is None:
            stride = combnet.HOPSIZE

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=stride,
            stride=stride,
        )
        self.filters = torch.nn.Sequential(
            C2(
                1,
                n_filters,
                sr=combnet.SAMPLE_RATE,
                **comb_kwargs,
            ),
        )
        if "min_freq" not in comb_kwargs:
            self.filters[0].f.data = centers[:n_filters, None]
            if "alpha" in comb_kwargs and comb_kwargs["alpha"] < 0:
                self.filters[0].f.data *= 2

        # activation = torch.nn.ReLU
        activation = torch.nn.ELU

        if linear_channels is None:
            linear_channels = n_filters * 8

        self.layers = torch.nn.Sequential(
            *(
                [
                    torch.nn.Conv2d(1, 8, (5, 5), (1, 1), (2, 2)),
                    activation(),
                ]
                + sum(
                    [
                        [
                            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (2, 2)),
                            activation(),
                        ]
                        for _ in range(1, n_conv_layers)
                    ],
                    start=[],
                )
                + [
                    torch.nn.Flatten(1, 2),
                    Permute(0, 2, 1),
                    torch.nn.Linear(linear_channels, 48),
                    activation(),
                    Permute(0, 2, 1),
                    torch.nn.AdaptiveAvgPool1d(1),
                    activation(),
                    torch.nn.Flatten(1, 2),
                    torch.nn.Linear(48, 24),
                    torch.nn.Softmax(dim=1),
                ]
            )
        )
        self.register_buffer("window", torch.hann_window(combnet.WINDOW_SIZE))

    def _extract_features(self, audio):
        features = self.filters(audio)
        if combnet.COMB_ACTIVATION is not None:
            features = combnet.COMB_ACTIVATION(features)
        return self.max_pool(features)

    def parameter_groups(self):
        groups = {}
        groups["f0"] = [self.filters[0].parametrizations.f.original]
        groups["main"] = list(self.layers.parameters())  # + [self.filters[0].a]
        return groups

    def forward(self, audio):
        features = self._extract_features(audio)
        return self.layers(features.unsqueeze(1))
