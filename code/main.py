import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_data, input_type_generator, metric_best_comparator, model_definer, metric_json_filepath_maker
from train import TrainSTT
import arg_parse_setup
import start_visualization_server
from warning_suppression import warning_suppression


def main(passed_args, metric_filepath):

    input_type = input_type_generator(passed_args.model)

    train_loader, test_loader = get_data(
        train_json_path='/stt/dataset_collection/keren_or/kerenor_train.json',
        valid_json_path='/stt/dataset_collection/keren_or/kerenor_valid.json',
        batch_size=passed_args.batch_size,
        input_type=input_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_definer(passed_args.model, device)

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

    for epoch in range(passed_args.epochs):
        print(f'\n\nEpoch[{epoch + 1}/{passed_args.epochs}]')
        trainer.train(model, device, train_loader, criterion, optimizer, scheduler, epoch, passed_args.model,
                      metric_filepath, passed_args.batch_size)
        wer_test = trainer.test(model, device, test_loader, criterion, epoch, passed_args.model, metric_filepath)

        best_wer = metric_best_comparator(wer_test, best_wer, 'WER', model, passed_args.model, num_parameters,
                                          args.model_save_path)


if __name__ == '__main__':
    args = arg_parse_setup.parse_args()
    warning_suppression()

    desired_metric_json_filepath = metric_json_filepath_maker(args)
    start_visualization_server.start_visualization(
        filepath=desired_metric_json_filepath,
        update_time_seconds=10,
        model_name=args.model
    )

    main(args, desired_metric_json_filepath)
