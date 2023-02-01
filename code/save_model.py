import torch
from deepspeech.model import DeepSpeech


model = DeepSpeech(
        n_cnn_layers=3,
        n_rnn_layers=5,
        rnn_dim=512,
        n_class=30,
        n_feats=128,
        stride=2,
        dropout=0.1)
model.load_state_dict(torch.load('deepspeech_kerenor_23705886.pt', map_location=torch.device('cpu'))['model_state_dict'])
model.eval()
model_jit = torch.jit.script(model)
model_jit.save('deepspeech_kerenor_model_23705886.pt')
# sample = torch.rand((64, 1, 128, 411))
# traced_model = torch.jit.trace(model, sample)
# torch.jit.save(traced_model, "deepspeech_kerenor_model_23705886.pt")
print('done')

