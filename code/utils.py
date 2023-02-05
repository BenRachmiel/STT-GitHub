import numpy as np
import torch
import torch.nn as nn
import torchaudio
import random
import json
from scipy.io import wavfile
import sys


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein distance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein distance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: reference sentence
    :type reference: basestring
    :param hypothesis: hypothesis sentence
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: reference sentence
    :type reference: basestring
    :param hypothesis: hypothesis sentence
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words substituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: reference sentence
    :type reference: basestring
    :param hypothesis: hypothesis sentence
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer_value = float(edit_distance) / ref_len
    return wer_value


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate character error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: reference sentence
    :type reference: basestring
    :param hypothesis: hypothesis sentence
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        print(reference)
        raise ValueError("Length of reference should be greater than 0.")

    cer_value = float(edit_distance) / ref_len
    return cer_value


class STTDataLoader:
    def __init__(self, json_path, batch_size, transform, max_duration=float('inf'), min_duration=0, shuffle=False):
        # initialization of loader, passing args to object
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.shuffle = shuffle
        self.data_pointers = self._read_content(json_path)
        self.batch_size = batch_size
        self.index = 0
        self.iterations = int(len(self.data_pointers) // self.batch_size)
        self.transform = transform
        self.current_lines = None
        self.current_batch = None

    def __len__(self):
        return len(self.data_pointers)

    def __next__(self):
        # iterator replacement
        if self.index < self.iterations:
            self.current_lines = self.data_pointers[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            batch = self._fetch_batch(self.current_lines)
            self.current_batch = self._process_batch(batch)
            self.index += 1
            return self.current_batch
        elif self.index == self.iterations:
            self.current_lines = self.data_pointers[self.index * self.batch_size:]
            if len(self.current_lines) != 0:
                batch = self._fetch_batch(self.current_lines)
                self.current_batch = self._process_batch(batch)
                self.index += 1
                return self.current_batch
            else:
                self.index = 0
                if self.shuffle:
                    random.shuffle(self.data_pointers)
                raise StopIteration
        else:
            self.index = 0
            if self.shuffle:
                random.shuffle(self.data_pointers)
            raise StopIteration

    def __iter__(self):
        # here for 'formality'
        return self

    def _read_content(self, json_path):
        def is_remove_file(line):
            # checking for bad data, i.e. duration out of range, text empty
            line_dict = json.loads(line)
            if line_dict['duration'] >= self.max_duration or line_dict['duration'] < self.min_duration:
                return True
            elif len(line_dict['text']) == 0:
                return True
            else:
                return False

        with open(json_path, mode='r', encoding="utf-8-sig") as file:
            content = file.readlines()

        content = [line for line in content if not is_remove_file(line)]
        if self.shuffle:
            random.shuffle(content)
        return content

    @staticmethod
    def _fetch_batch(lines):
        batch = []
        for line in lines:
            p = str(line)
            data = json.loads(p)
            file_path = data['audio_filepath']
            text = data['text']
            samplerate, waveform = wavfile.read(file_path)
            waveform = np.resize(waveform, (len(waveform), 1)).T
            if waveform.dtype == 'int16':
                waveform = normalize_int16(waveform)
            waveform = torch.from_numpy(waveform)
            triade = (waveform.type(torch.float), 'a', text, 0, 0, 0)
            batch.append(triade)
        return batch

    def _process_batch(self, batch):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        text_transform = TextTransform()
        for waveform, _, utterance, _, _, _ in batch:
            spec = self.transform(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0] // 2)
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return spectrograms, labels, input_lengths, label_lengths


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        א 2
        ב 3
        ג 4
        ד 5
        ה 6
        ו 7
        ז 8
        ח 9
        ט 10
        י 11
        כ 12
        ך 13
        ל 14
        מ 15
        ם 16
        נ 17
        ן 18
        ס 19
        ע 20
        פ 21
        ף 22
        צ 23
        ץ 24
        ק 25
        ר 26
        ש 27
        ת 28
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to a text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


def normalize_int16(signal):
    signal = signal.astype(np.float32)
    return signal / 32768.0


def get_data(train_json_path, valid_json_path, batch_size, input_type='melspectrogram'):
    if input_type == 'melspectrogram':
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=8000),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
        valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=8000)

    if input_type == 'MFCC':
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MFCC(
                sample_rate=8000,
                n_mfcc=128,
                dct_type=2),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
        valid_audio_transforms = torchaudio.transforms.MFCC(
            sample_rate=8000,
            n_mfcc=128,
            dct_type=2
        )

    train_loader = STTDataLoader(train_json_path, batch_size, train_audio_transforms, shuffle=True)
    test_loader = STTDataLoader(valid_json_path, batch_size, valid_audio_transforms, shuffle=True)
    return train_loader, test_loader


def input_type_generator(model):
    input_type = 'ERROR'
    if model == 'Wav2Letter':
        input_type = 'MFCC'
    if model == 'DeepSpeech':
        input_type = 'melspectrogram'
    return input_type
