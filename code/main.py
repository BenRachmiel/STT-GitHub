import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import get_data, input_type_generator
from deepspeech.model import DeepSpeech
from train import TrainSTT
import arg_parse_setup
import torchaudio
import start_visualization_server
from warning_suppression import warning_suppression


def main(passed_args):

    # TODO: Add validation to training

    input_type = input_type_generator(passed_args.model)

    train_loader, test_loader = get_data(
        train_json_path='/home/yakir/PycharmProjects/stt-plus-visualization/venv/dataset_collection/kerenor_train.json',
        valid_json_path='/home/yakir/PycharmProjects/stt-plus-visualization/venv/dataset_collection/kerenor_valid.json',
        batch_size=passed_args.batch_size,
        input_type=input_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if passed_args.model == 'DeepSpeech':
        model = DeepSpeech(
            n_cnn_layers=3,
            n_rnn_layers=5,
            rnn_dim=512,
            n_class=30,
            n_feats=128,
            stride=2,
            dropout=0.1
        ).to(device)
    if passed_args.model == 'Wav2Letter':
        model = torchaudio.models.Wav2Letter(input_type='mfcc', num_features=128, num_classes=30).to(device)

    num_parameters = sum([param.nelement() for param in model.parameters()])
    print(f'Running {passed_args.model} with {num_parameters} parameters')

    optimizer = optim.AdamW(model.parameters(), passed_args.learning_rate)
    criterion = nn.CTCLoss(blank=29).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=passed_args.learning_rate,
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=passed_args.epochs,
                                              anneal_strategy='linear')

    trainer = TrainSTT()

    best_wer = float('inf')

    epoch_array = np.array(range(passed_args.epochs))
    wer_array = np.array([])
    for epoch in range(passed_args.epochs):
        print(f'\n\nEpoch[{epoch + 1}/{passed_args.epochs}]')
        trainer.train(model, device, train_loader, criterion, optimizer, scheduler, epoch, passed_args.model)
        wer_test = trainer.test(model, device, test_loader, criterion, epoch, passed_args.model)
        np.append(wer_array, wer_test)  # for plotting
        if wer_test < best_wer:
            best_wer = wer_test
            print(f'saving model with wer = {wer_test}')
            torch.save({'model_state_dict': model.state_dict()}, f"{passed_args.model}_ko_22_08_2022_{num_parameters}.pt")


if __name__ == '__main__':
    args = arg_parse_setup.parse_args()

    warning_suppression()

    start_visualization_server.start_visualization(
        filepath='visualization_data.json',
        update_time_seconds=10,
        model_name=args.model
    )

    main(args)
