# coding=utf-8
import torch
import torch.nn as nn
import torchaudio
from deepspeech.model import DeepSpeech
import numpy as np
from scipy.io import wavfile
from utils import normalize_int16
import torch.nn.functional as F
from utils import TextTransform
from utils import _levenshtein_distance

import time


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
    model.load_state_dict(torch.load('deepspeech_rapat_23705886.pt', map_location=torch.device('cpu'))['model_state_dict'])
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


def correct_sentence(sentence, bow):
    sentence_list = sentence.split(' ')
    correct = ''
    for pred_word in sentence_list:
        if not (pred_word == '' or pred_word == ' '):
            scores = []
            for word in bow:
                scores.append(_levenshtein_distance(word, pred_word))
            correct += bow[scores.index(min(scores))]
            correct += ' '
    return correct


def main():

    BOW = list({0: 'סימון', 1: 'צוות', 2: 'אני', 3: 'ירי', 4: 'למארזים', 5: 'הורד', 6: 'חדל', 7: 'במענק', 8: 'פגזים', 9: 'כיוון',
     10: 'העבר', 11: 'הזן', 12: 'לבזנט', 13: 'קודקוד', 14: 'לי', 15: 'מהר', 16: 'אחורית', 17: 'חיר', 18: 'פעמיים',
     19: 'אתה', 20: 'אחרון', 21: 'שנייה', 22: 'סוף', 23: 'מיסוך', 24: 'תותח', 25: 'נונטת', 26: 'לפח', 27: 'בשלי',
     28: 'מרכז', 29: 'טנקים', 30: 'חום', 31: 'כוונות', 32: 'קצר', 33: 'עד', 34: 'בחצב', 35: 'וימינה', 36: 'בגיזרתכם',
     37: 'תותחן', 38: 'ארוך', 39: 'הבא', 40: 'מדידה', 41: 'בועית', 42: 'חץ', 43: 'זהירות', 44: 'הקרובים', 45: 'בקצה',
     46: 'בחץ', 47: 'הכלים', 48: 'המשך', 49: 'יורה', 50: 'טען', 51: 'נגמש', 52: 'לזירות', 53: 'נפיץ', 54: 'הושמד',
     55: 'היכונו', 56: 'עקיבה', 57: 'גימל', 58: 'זמנים', 59: 'נהג', 60: 'עלייה', 61: 'מהבזנט', 62: 'את', 63: 'אחורה',
     64: 'דווחו', 65: 'ימני', 66: 'מאוזנת', 67: 'מג', 68: 'איתור', 69: 'התקן', 70: 'הפסק', 71: 'בכל', 72: 'אפיסקופים',
     73: 'לנגמש', 74: 'ימינה', 75: 'לעבר', 76: 'כאן', 77: 'שאינו', 78: 'וודא', 79: 'נבחרת', 80: 'ממתין', 81: 'תצפית',
     82: 'לטווחים', 83: 'צרור', 84: 'עצור', 85: 'מרגמה', 86: 'להתקפה', 87: 'מימין', 88: 'זוהר', 89: 'עשן', 90: 'במצלמה',
     91: 'לעמדה', 92: 'כוונת', 93: 'סריקה', 94: 'שמאלה', 95: 'אוייב', 96: 'שמאלי', 97: 'לדיווח', 98: 'הפעל', 99: 'יעד',
     100: 'המבנה', 101: 'בכינון', 102: 'השמד', 103: 'התותחן', 104: 'גזרה', 105: 'בצע', 106: 'בטווח', 107: 'טילים',
     108: 'הוצא', 109: 'מקביל', 110: 'סטטוס', 111: 'קדימה', 112: 'מטרה', 113: 'פלס', 114: 'צמד', 115: 'הוסף',
     116: 'לכד', 117: 'מפקד', 118: 'דרוס', 119: 'אש', 120: 'פלוס', 121: 'השמדה', 122: 'רכב', 123: 'בהיחשפות',
     124: 'הזמינים', 125: 'מהשיח'}.values())

    device = torch.device("cpu")
    model = load_model(device)
    model.eval()

    for _ in range(5):
        folder = np.random.randint(1, 11)
        file = np.random.randint(1, 301)
        path = f"D:/pack/nemo_project/dataset_collection/raphat/transcription_list{folder}/{file}.WAV.wav"
        print(f"transcription_list{folder}/{file}.WAV.wav")
        data = load_data(path, device)

        t0 = time.time()
        output = transcribe_file(model, data)
        # print(output)
        print(time.time() - t0)
        output = correct_sentence(output, BOW)
        # print(output)

        with open('transcriptions.txt', mode='a', encoding="utf-8-sig") as file:
            file.write(output)
            file.write('\n')


if __name__ == '__main__':
    main()




