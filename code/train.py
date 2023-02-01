import torch
import torch.nn.functional as F
from utils import cer, wer, TextTransform
import numpy as np
import json


class TrainSTT:

    @staticmethod
    def greedy_decoder(output, labels, label_lengths, blank_label=29, collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        text_transform = TextTransform()
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(text_transform.int_to_text(decode))
        return decodes, targets

    def train(self,
              model,
              device,
              train_loader,
              criterion,
              optimizer,
              scheduler,
              epoch,
              model_type):

        model.train()
        running_loss = 0.0
        running_cer = 0.0
        running_wer = 0.0
        print(f'\tTRAIN')
        for batch_idx, data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # TODO clean this up
            optimizer.zero_grad()
            if model_type == 'Wav2Letter':
                spectrograms = np.squeeze(spectrograms, axis=1)
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)
            if model_type == 'Wav2Letter':
                output = output.transpose(0, 2)

            loss = criterion(output, labels, input_lengths, label_lengths)
            running_loss += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            scheduler.step()

            decoded_preds, decoded_targets = self.greedy_decoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                running_cer += cer(decoded_targets[j], decoded_preds[j])
                running_wer += wer(decoded_targets[j], decoded_preds[j])
        print(f'\t\ttarget:\t{decoded_targets[-1]}')
        print(f'\t\tprediction:\t{decoded_preds[-1]}')
        return

    def test(
            self,
            model,
            device,
            test_loader,
            criterion,
            epoch,
            model_type,
            filepath):

        model.eval()
        running_loss = 0.0
        running_cer = 0.0
        running_wer = 0.0
        print(f'\tEVAL')
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = data
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                if model_type == 'Wav2Letter':
                    spectrograms = np.squeeze(spectrograms, axis=1)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)
                if model_type == 'Wav2Letter':
                    output = output.transpose(0, 2)

                loss = criterion(output, labels, input_lengths, label_lengths)
                running_loss += loss.item()

                decoded_preds, decoded_targets = self.greedy_decoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    running_cer += cer(decoded_targets[j], decoded_preds[j])
                    running_wer += wer(decoded_targets[j], decoded_preds[j])
        print(f'\t\ttarget:\t{decoded_targets[-1]}')
        print(f'\t\tprediction:\t{decoded_preds[-1]}')

        # for visualization testing, TODO : move to separate function
        with open(filepath) as json_file:
            data = json.load(json_file)
        data['Epoch'].append(epoch)
        data['Loss'].append(running_loss / batch_idx)
        data['Loss2'].append(running_loss / batch_idx)
        data['WER'].append(running_wer / len(test_loader))
        data['CER'].append(running_cer / len(test_loader))
        with open(filepath, 'w') as fp:
            fp.seek(0)
            json.dump(data, fp)

        return running_wer / len(test_loader)
