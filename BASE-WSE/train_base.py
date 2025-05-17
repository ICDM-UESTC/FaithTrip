import numpy as np
import torch
import argparse
import time
import random
from tqdm import tqdm
from traj_dataset import TrajectoryDataset
from utils import load_data, load_flickr_data, split_df_train_eval_test, split_df_leave_one_out
from traj_base_trainer import TrajBaseTrainer

"""
This model (T-Base-WSE) is a naive transformer-like model for trip planning, multiple steps
(1) load dataset and process data, including renumber, padding, etc.
(2) build basic model (based on transformer architecture)
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
        all_alt_f1_list = []
        all_alt_pairs_f1_list = []
        all_total_f1_list = []
        all_total_pairs_f1_list = []
        all_mean_dis_list = []
        all_distortion_dis_list = []
        # Load and process data
        df, poi_dis_dict = load_flickr_data(dataset=opt.dataset)
        # counting the process time
        start_time = time.time()
        # Leave_one_out
        for split_index in tqdm(range(df.shape[0])):
            print(f'total trajectory is {df.shape[0]}, current index is {split_index}')

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
            trainer = TrajBaseTrainer(train_dataset=train_dataset,
                                      eval_dataset=test_dataset,
                                      poi_dis_dict=poi_dis_dict,
                                      weight_num=opt.se_weight,
                                      lr=opt.lr,
                                      data_type=opt.dataset_type,
                                      batch_size=opt.batch_size,
                                      d_model=opt.d_model,
                                      num_encoder_layers=opt.num_encoder_layers,
                                      num_epochs=opt.num_epochs,
                                      venue_vocab_size=venue_vocab_size,
                                      hour_vocab_size=hour_vocab_size,
                                      max_length_venue_id=max_length_venue_id)

            start, end, alt_f1, alt_pairs_f1, total_f1, total_pairs_f1, mean, distortion = trainer.train()

            all_start_acc_list.append(start)
            all_end_acc_list.append(end)
            all_alt_f1_list.append(alt_f1)
            all_alt_pairs_f1_list.append(alt_pairs_f1)
            all_total_f1_list.append(total_f1)
            all_total_pairs_f1_list.append(total_pairs_f1)
            all_mean_dis_list.append(mean)
            all_distortion_dis_list.append(distortion)

        # process distance：
        all_mean_dis_list = np.array(all_mean_dis_list).flatten()
        # error_traj_counts = len(all_mean_dis_list) - np.count_nonzero(all_mean_dis_list != -1)
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
        alt_f1 = np.mean(all_alt_f1_list)
        alt_pairs_f1 = np.mean(all_alt_pairs_f1_list)
        total_f1 = np.mean(all_total_f1_list)
        total_pairs_f1 = np.mean(all_total_pairs_f1_list)

        print(f'max f1 score: {alt_f1}, max pairs_f1: {alt_pairs_f1}')
        print(f'total f1 score: {total_f1}, total pairs_f1: {total_pairs_f1}')

        # mean_dis and distortion_dis
        mean_dis = np.mean(non_error_dis_list)
        # distortion_dis = np.mean(all_distortion_dis_list)
        print(f'total_mean_distance: {mean_dis}')

        with open(
                f'./results/BASE-WSE/{opt.dataset}/{opt.lr}-{opt.batch_size}-{opt.d_model}-{opt.num_encoder_layers}-{opt.num_epochs}-{opt.se_weight}.txt',
                'a') as file:

            file.write('===' * 18 + '\n')
            file.write(f'max f1-score: {alt_f1}' + '\n')
            file.write(f'max pairs f1-score: {alt_pairs_f1}' + '\n')
            file.write(f'total f1-score: {total_f1}' + '\n')
            file.write(f'total pairs f1-score: {total_pairs_f1}' + '\n')
            file.write(f'start ratio: {start_acc}' + '\n')
            file.write(f'end ratio: {end_acc}' + '\n')
            file.write(f'mean_distance: {mean_dis}' + '\n')
            file.write(f'total process times: {final_time}' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help='manual seed')
    parser.add_argument('--dataset', type=str, default='Osak', help='dataset')
    parser.add_argument('--dataset_type', type=str, default='Flickr', help='dataset_type')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # 0.001
    parser.add_argument('--se_weight', type=int, default=5, help='the weight to count CeLoss in start and end')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--num_encoder_layers', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=20, help='number of epoch')

    args = parser.parse_args()

    main(args)
