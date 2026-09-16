"""Compare compiled/eager outputs and gradients before an aligned GPU run."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from train_srnn.config import compose_config
from train_srnn.models.factory import build_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg = compose_config(['task=synthetic', 'model.num_units=32',
                          'model.variants=[srnn-no-adapt,srnn-sfa1-std1,srnn-sfa3-std2]',
                          'model.variant_seeds=[1,2,3]'])
    eager = build_model(cfg).cell.cuda()
    compiled_base = copy.deepcopy(eager)
    compiled = torch.compile(compiled_base)
    state = eager.init_state(2).detach()
    inputs = torch.randn(2, 4, device='cuda')
    def run(cell):
        y, s = cell(inputs, state)
        loss = y.square().sum() + s.square().sum() / s.numel()
        loss.backward()
        return y.detach(), s.detach()
    reference = run(eager)
    actual = run(compiled)
    errors = {}
    for label, a, b in zip(('output', 'state'), actual, reference):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        errors[label] = (a-b).abs().max().item()
    for (name, p), (_, q) in zip(eager.named_parameters(), compiled_base.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
            continue
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)
        errors['gradient:' + name] = (p.grad-q.grad).abs().max().item()
    print(json.dumps({'torch': torch.__version__, 'compiled_eager_max_abs': errors}, indent=2))


if __name__ == '__main__':
    main()
