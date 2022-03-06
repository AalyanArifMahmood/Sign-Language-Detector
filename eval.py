from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import numpy as np

import onnx

from train import Neural


def evaluate(outputs: Variable, labels: Variable) -> float:
    Y = labels.numpy()
    Yhat = np.argmax(outputs, axis=1)
    return float(np.sum(Yhat == Y))


def batch_evaluate(
        net: Neural,
        dataloader: torch.utils.data.DataLoader) -> float:
    score = n = 0.0
    for batch in dataloader:
        n += len(batch['image'])
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])
    return score / n


def validate():
    net = Neural().float().eval()

    pretrained_model = torch.load("checkpoint.pth")
    net.load_state_dict(pretrained_model)

    fname = "signlanguage.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])

    model = onnx.load(fname)
    onnx.checker.check_model(model)


def main():
    validate()


if __name__ == '__main__':
    main()
