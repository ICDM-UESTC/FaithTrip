import torch
import argparse
import time
import wandb
import random
import numpy as np
from tqdm import tqdm
from traj_dataset import TrajectoryDataset
from utils import load_data, load_flickr_data, split_df_train_eval_test, split_df_leave_one_out
from traj_diffusion_trainer import TrajDiffusionTrainer

"""
This model (diffusion) is a naive transformer-diffusion model for trip planning, multiple steps
(1) load dataset and process data, including renumber, padding, etc.
(2) build basic model (based on transformer architecture)
(3) add diffusion model and classifier
"""


def main(opt):
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    assert opt.dataset_type in ['Gowalla', 'Flickr']

    if opt.dataset_type == 'Flickr':

        # define the calculator list
        all_start_acc_list = []
        all_end_acc_list = []
        all_f1_list = []
        all_pairs_f1_list = []
        all_mean_dis_list = []
        all_distortion_dis_list = []
        # Load and process data and location dict
        df, poi_dis_dict = load_flickr_data(dataset=opt.dataset)
        # counting the process time
        start_time = time.time()
        # Leave_one_out
        for split_index in tqdm(range(df.shape[0])):
            print(f'total trajectory is {df.shape[0]}, current index is {split_index}')
            if opt.wandb:
                wandb.init(project=f'DiffTrip-Flickr-{args.dataset}',
                           name=f'city-{args.dataset}-seed-{args.seed}-batch-{args.batch_size}-epoch-'
                                f'{args.num_epochs}-fold-{split_index}')

            train_df, test_df = split_df_leave_one_out(split_index, df, opt.seed)
            train_dataset = TrajectoryDataset(train_df)
            test_dataset = TrajectoryDataset(test_df)

            # initiate trainer
            venue_vocab_size = df['venue_ID'].explode().nunique() + 1
            hour_vocab_size = 24 + 1
            # Convert the venue_ID column to a list of lists
            venue_ids_lists = df["venue_ID"].tolist()
            # Calculate the maximum length of venue_ID
            max_length_venue_id = max(len(venue_ids) for venue_ids in venue_ids_lists)
            trainer = TrajDiffusionTrainer(train_dataset=train_dataset,
                                           eval_dataset=test_dataset,
                                           poi_dis_dict=poi_dis_dict,
                                           data_type=opt.dataset_type,
                                           wandb_or_not=opt.wandb,
                                           step=opt.step,
                                           max_noise=opt.max_noise,
                                           lr=opt.lr,
                                           batch_size=opt.batch_size,
                                           num_epochs=opt.num_epochs,
                                           d_model=opt.d_model,
                                           num_encoder_layers=opt.num_encoder_layers,
                                           venue_vocab_size=venue_vocab_size,
                                           hour_vocab_size=hour_vocab_size,
                                           max_length_venue_id=max_length_venue_id,
                                           fast_sampling = opt.sampling)

            start, end, f1, pairs_f1, mean, distortion = trainer.train()

            all_start_acc_list.append(start)
            all_end_acc_list.append(end)
            all_f1_list.append(f1)
            all_pairs_f1_list.append(pairs_f1)
            all_mean_dis_list.append(mean)
            all_distortion_dis_list.append(distortion)

            if opt.wandb:
                wandb.finish()

        # process distance：
        all_mean_dis_list = np.array(all_mean_dis_list).flatten()
        error_traj_counts = len(all_mean_dis_list) - np.count_nonzero(all_mean_dis_list != -1)
        non_error_dis_list = all_mean_dis_list[all_mean_dis_list != -1]

        end_time = time.time()
        final_time = end_time - start_time
        print("the running time：%.2f seconds." % final_time)

        print("===" * 18)
        print("the final results...")
        # start and end accuracy
        start_acc = np.sum(all_start_acc_list) / df.shape[0]
        end_acc = np.sum(all_end_acc_list) / df.shape[0]
        print(f'start accuracy: {start_acc}, end accuracy: {end_acc}')
        # f1 and pairs_f1
        f1 = np.mean(all_f1_list)
        f1_std = np.std(all_f1_list)
        pairs_f1 = np.mean(all_pairs_f1_list)
        pairs_f1_std = np.std(all_pairs_f1_list)
        print(f'mean: f1 score: {f1}, pairs_f1: {pairs_f1}')
        print(f'std: f1 score: {f1_std}, pairs_f1: {pairs_f1_std}')
        # mean_dis and distortion_dis
        mean_dis = np.mean(non_error_dis_list)
        # distortion_dis = np.mean(all_distortion_dis_list)
        print(f'total_mean_distance: {mean_dis}')

        if opt.wandb:
            wandb.finish()

        # save the results:
        with open(
                f'./results/FaithTrip/{opt.dataset}/{opt.lr}-{opt.batch_size}-{opt.d_model}-{opt.num_encoder_layers}-{opt.num_epochs}.txt',
                'a') as file:
            file.write('===' * 18 + '\n')
            file.write(f'{opt.sampling}-step!!!' + '\n')
            file.write(f'seed:{opt.seed}, step:{opt.step}, noise:{opt.max_noise}' + '\n')

            file.write(f'total f1-score: {f1}' + '\n')
            file.write(f'total pairs f1-score: {pairs_f1}' + '\n')
            file.write(f'start ratio: {start_acc}' + '\n')
            file.write(f'end ratio: {end_acc}' + '\n')
            file.write(f'mean_distance: {mean_dis}'+ '\n')
            file.write(f'total process times: {final_time}' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # setting
    parser.add_argument('--seed', type=int, default=2024, help='manual seed')
    parser.add_argument('--wandb', type=bool, default=False, help='use wandb or not')
    parser.add_argument('--dataset', type=str, default='Osak', help='dataset')
    parser.add_argument('--dataset_type', type=str, default='Flickr', help='dataset_type')
    parser.add_argument('--sampling', type=int, default=4, help='fast sampling step.')
    # train:
    parser.add_argument('--step', type=int, default=32, help='diffusion_step')  # 32
    parser.add_argument('--max_noise', type=float, default=0.02, help='beta_t')  # 0.02
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # 0.01
    parser.add_argument('--batch_size', type=int, default=4)  # 4
    parser.add_argument('--d_model', type=int, default=16)  # 32/64
    parser.add_argument('--num_encoder_layers', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=20, help='number of epoch')

    args = parser.parse_args()

    main(args)
