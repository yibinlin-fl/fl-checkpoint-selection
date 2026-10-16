import torch
from p1.models import build_model
from p1.training import fedavg


def test_model_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(11)
    model = build_model('small_cnn')
    x = torch.randn(2,3,32,32)
    model.eval()
    expected = model(x).detach().clone()
    path = tmp_path / 'checkpoint.pt'
    torch.save(model.state_dict(), path)
    restored = build_model('small_cnn')
    restored.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
    restored.eval()
    assert torch.equal(expected, restored(x).detach())


def test_fedavg_weighted_mean():
    states=[{'w':torch.tensor([1.0])},{'w':torch.tensor([5.0])}]
    got=fedavg(states,[1,3])
    assert torch.isclose(got['w'],torch.tensor([4.0])).all()
