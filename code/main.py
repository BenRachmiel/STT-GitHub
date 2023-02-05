import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_data, input_type_generator, metric_best_comparator, model_definer, filepath_maker
from train import TrainSTT
import arg_parse_setup
import start_visualization_server
from warning_suppression import warning_suppression


def main(passed_args, start_time):
    input_type = input_type_generator(passed_args.model)

    train_loader, test_loader = get_data(
        train_json_path=args.train_data_json_path,
        valid_json_path=args.valid_data_json_path,
        batch_size=passed_args.batch_size,
        input_type=input_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO, for later, model loading + displaying prev data from model
    model = model_definer(passed_args.model, device)

    num_parameters = sum([param.nelement() for param in model.parameters()])
    print(f'Running {passed_args.model} with {num_parameters} parameters')

    # TODO optimizer and criterion in argparse
    optimizer = optim.AdamW(model.parameters(), passed_args.learning_rate)
    criterion = nn.CTCLoss(blank=29).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=passed_args.learning_rate,
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=passed_args.epochs,
                                              anneal_strategy='linear')

    trainer = TrainSTT()

    best_wer = float('inf')
    path_to_metric_json = passed_args.metric_filepath + \
                          f'/{passed_args.model}_run_{start_time.hour}_{start_time.minute}_metrics.json'

    for epoch in range(passed_args.epochs):
        print(f'\n\nEpoch[{epoch + 1}/{passed_args.epochs}]')

        trainer.train(model, device, train_loader, criterion, optimizer, scheduler, epoch, passed_args.model,
                      path_to_metric_json, passed_args.batch_size)

        # TODO not train, validate
        wer_test = trainer.validate(model, device, test_loader, criterion, epoch, passed_args.model,
                                    path_to_metric_json)

        best_wer = metric_best_comparator(wer_test, best_wer, 'WER', model, passed_args.model, num_parameters,
                                          args.model_save_path)

    # TODO, after train and validate, test model on new data


if __name__ == '__main__':
    args = arg_parse_setup.parse_args()
    warning_suppression()

    now = filepath_maker(args.metric_filepath, args.model_save_filepath, args.model)

    start_visualization_server.start_visualization(
        filepath=args.metric_filepath + f'/{args.model}_run_{now.hour}_{now.minute}_metrics.json',
        model_name=args.model
    )

    main(args, now)
