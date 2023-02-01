import torch
import torch.nn as nn
import torchaudio
from deepspeech.model import DeepSpeech
import numpy as np
from scipy.io import wavfile
from utils import normalize_int16
import torch.nn.functional as F
from utils import TextTransform


def greedy_decoder(output, blank_label=29, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2).T
    text_transform = TextTransform()
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
    return text_transform.int_to_text(decode)


def load_model(device):
    model = DeepSpeech(
        n_cnn_layers=3,
        n_rnn_layers=5,
        rnn_dim=512,
        n_class=30,
        n_feats=128,
        stride=2,
        dropout=0.1).to(device)
    model.load_state_dict(torch.load('deepspeech_kerenor_22_7_2022_23705886.pt', map_location=torch.device('cpu'))['model_state_dict'])
    return model


def load_data(path, device):
    data = []
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=8000)
    samplerate, waveform = wavfile.read(path)
    waveform = np.resize(waveform, (len(waveform), 1)).T
    waveform = normalize_int16(waveform)
    waveform = torch.from_numpy(waveform)
    spec = transform(waveform).squeeze(0).transpose(0, 1)
    data.append(spec)
    data = nn.utils.rnn.pad_sequence(data, batch_first=True).unsqueeze(1).transpose(2, 3)
    data = data.to(device)
    return data


def transcribe_file(model, data):
    output = model(data)
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1)  # (time, batch, n_class)
    output = greedy_decoder(output)
    return output


def main():
    device = torch.device("cpu")
    model = load_model(device)

    folder = np.random.randint(1, 11)
    file = np.random.randint(1, 301)
    path = f"D:/pack/nemo_project/dataset_collection/kerenor_train/train/recordings/Recording #11.wav"
    print(f"transcription_list{folder}/{file}.WAV.wav")
    data = load_data(path, device)

    output = transcribe_file(model, data)
    print(output)


if __name__ == '__main__':
    main()
