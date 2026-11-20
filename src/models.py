"""
Módulo de arquitetura de Redes Neurais Artificiais (MLP parametrizável em PyTorch)
para os problemas de Regressão (tempo de entrega) e Classificação (atraso na entrega).
"""

from typing import Iterable, Tuple, Union
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Perceptron Multicamadas (MLP) parametrizável em PyTorch.

    Suporta variação dinâmica de:
    - número de camadas ocultas e neurônios por camada (`hidden`);
    - função de ativação (`ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`, `ELU`, `GELU`);
    - regularização por `Dropout` (`dropout`);
    - normalização em lote (`use_batchnorm`);
    - estratégia de inicialização de pesos (`default`, `xavier`, `he`, `zeros`).
    - camada de saída linear sem ativação (1 neurônio para regressão ou logit binário).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden: Union[Tuple[int, ...], Iterable[int], int] = (32, 32),
        activation: str = "ReLU",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        weight_init: str = "default",
    ):
        super().__init__()
        if isinstance(hidden, int):
            hidden_tuple = (hidden,)
        else:
            hidden_tuple = tuple(int(h) for h in hidden)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = hidden_tuple
        self.activation_name = str(activation)
        self.dropout_rate = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.weight_init = str(weight_init)

        act_cls = self._resolve_activation(self.activation_name)
        camadas = []
        d = self.in_dim

        for h in self.hidden:
            linear_layer = nn.Linear(d, h)
            camadas.append(linear_layer)
            if self.use_batchnorm:
                camadas.append(nn.BatchNorm1d(h))
            camadas.append(act_cls())
            if self.dropout_rate > 0.0:
                camadas.append(nn.Dropout(self.dropout_rate))
            d = h

        # Camada de saída linear (sem função de ativação: valor contínuo na regressão ou logit na classificação)
        camadas.append(nn.Linear(d, self.out_dim))
        self.net = nn.Sequential(*camadas)

        if self.weight_init != "default":
            self._apply_weight_init(self.weight_init)

    @staticmethod
    def _resolve_activation(name: str):
        mapping = {
            "ReLU": nn.ReLU,
            "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01),
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
        }
        if name not in mapping:
            return getattr(nn, name)
        return mapping[name]

    def _apply_weight_init(self, mode: str) -> None:
        mode_lower = mode.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode_lower == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif mode_lower in ("he", "kaiming"):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif mode_lower == "zeros":
                    nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_config(self) -> dict:
        """Retorna dicionário serializável com a configuração completa da arquitetura."""
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "hidden": list(self.hidden),
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "use_batchnorm": self.use_batchnorm,
            "weight_init": self.weight_init,
        }


def count_trainable_parameters(model: nn.Module) -> int:
    """Calcula o número total de parâmetros treináveis (pesos e vieses) da rede."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_macs_per_sample(model: nn.Module) -> dict:
    """
    Calcula a complexidade estrutural da MLP:
    - número total de parâmetros treináveis;
    - número de camadas lineares (ocultas + saída);
    - operações de Multiplicação-Acumulação (MACs) por amostra na inferência.
    """
    n_linear = 0
    macs = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            n_linear += 1
            macs += m.in_features * m.out_features
    return {
        "n_parametros": count_trainable_parameters(model),
        "n_camadas_lineares": n_linear,
        "n_camadas_ocultas": max(0, n_linear - 1),
        "macs_por_amostra": macs,
    }
