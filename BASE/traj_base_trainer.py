import numpy as np
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from utils import *
from base_model import TrajectoryBaseModel
from rich.progress import track
from tqdm import tqdm


class TrajBaseTrainer:
    def __init__(self,
                 train_dataset,
                 eval_dataset,
                 poi_dis_dict,
                 lr,
                 data_type,
                 batch_size,
                 num_epochs,
                 d_model,
                 num_encoder_layers,
                 venue_vocab_size,
                 hour_vocab_size,
                 max_length_venue_id):

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.poi_dis_dict = poi_dis_dict
        self.data_type = data_type
        self.num_epochs = num_epochs
        self.venue_vocab_size = venue_vocab_size
        self.hour_vocab_size = hour_vocab_size
        self.max_length_venue_id = max_length_venue_id

        # build dataloader
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       collate_fn=collate_fn,
                                       shuffle=True,
                                       drop_last=True)

        self.eval_loader = DataLoader(eval_dataset,
                                      batch_size=1,
                                      collate_fn=collate_fn,
                                      drop_last=True)

        # build model
        self.model = TrajectoryBaseModel(venue_vocab_size, hour_vocab_size, max_length_venue_id,
                                         d_model=d_model,
                                         num_encoder_layers=num_encoder_layers
                                         ).cuda()

        # configuration
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index during loss calculation
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        epoch_start_list = []
        epoch_end_list = []
        epoch_alt_f1_score_list = []
        epoch_alt_pairs_f1_list = []
        epoch_total_f1_score_list = []
        epoch_total_pairs_f1_list = []
        epoch_distance_list = []
        epoch_distortion_distance_list = []

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            total_ids = 0
            correct_predictions = 0
            for masked_padded_venue_ids, masked_padded_hour_ids, padded_venue_ids, _, _ in \
                    track(self.train_loader, description=f"training {epoch} epoch..."):
                # print(masked_padded_venue_ids[0], masked_padded_hour_ids[0], padded_venue_ids[0])

                self.optimizer.zero_grad()
                venue_output = self.model(masked_padded_venue_ids.cuda(), masked_padded_hour_ids.cuda())
                venue_output = venue_output.cpu()

                loss = self.criterion(venue_output.view(-1, self.venue_vocab_size),
                                      padded_venue_ids.flatten())

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Get the predictions for the output venue IDs
                _, predicted_ids = torch.max(venue_output, dim=-1)
                predicted_ids.cpu()
                # print results
                # print(predicted_ids[0], padded_venue_ids[0])

                # Flatten the predictions and ground truth for comparison
                predicted_ids = predicted_ids.view(-1)
                venue_target = padded_venue_ids.flatten()

                # Exclude the padded values from calculations
                non_padded_indices = venue_target != 0
                venue_target = venue_target[non_padded_indices]
                predicted_ids = predicted_ids[non_padded_indices]

                # Count the number of correctly predicted masked IDs
                total_ids += venue_target.size(0)
                correct_predictions += (predicted_ids == venue_target).sum().item()

            epoch_loss = total_loss / len(self.train_loader)
            print(f'train loss: {epoch_loss}')

            # wandb.log({'train loss': epoch_loss, 'epoch': epoch_loss})

            precision = correct_predictions / total_ids
            print(f"train precision: {precision}")

            if self.data_type == 'Flickr':

                eval_loss, start_accuracy, end_accuracy, alt_f1, alt_pairs_f1, total_f1, total_pairs_f1, distance, \
                    distortion_distance = self.evaluate()
                epoch_start_list.append(start_accuracy)
                epoch_end_list.append(end_accuracy)
                epoch_alt_f1_score_list.append(alt_f1)
                epoch_alt_pairs_f1_list.append(alt_pairs_f1)
                epoch_total_f1_score_list.append(total_f1)
                epoch_total_pairs_f1_list.append(total_pairs_f1)
                epoch_distance_list.append(distance)
                epoch_distortion_distance_list.append(distortion_distance)

            else:
                eval_loss, start_ratio, end_ratio = self.evaluate()
                print(eval_loss)
        # calculate results
        f1_max_index = count_best_index(epoch_alt_f1_score_list, epoch_alt_pairs_f1_list, epoch_distance_list)
        f1_max_total_index = count_best_index(epoch_total_f1_score_list, epoch_total_pairs_f1_list, epoch_distance_list)
        # pick the max_f1 results in each epoch
        return epoch_start_list[f1_max_total_index], epoch_end_list[f1_max_total_index], \
            epoch_alt_f1_score_list[f1_max_index], epoch_alt_pairs_f1_list[f1_max_index], \
            epoch_total_f1_score_list[f1_max_total_index], epoch_total_pairs_f1_list[f1_max_total_index], \
            epoch_distance_list[f1_max_total_index], epoch_distortion_distance_list[f1_max_total_index]

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_ids = 0
        correct_predictions = 0

        # for the first venue
        total_first_venues = 0
        correct_predictions_first_venues = 0

        # for the last venue
        total_last_venues = 0
        correct_predictions_last_venues = 0

        # for the f1 and pairs f1
        alt_f1_list = []
        alt_pairs_f1_list = []
        total_f1_list = []
        total_pairs_f1_list = []
        # for the mean_dis and distor_dis
        mean_dis_list = []
        distor_dis_list = []

        with torch.no_grad():
            for masked_padded_venue_ids, masked_padded_hour_ids, padded_venue_ids, _, _ in \
                    track(self.eval_loader, description=f"evaluating..."):
                venue_output = self.model(masked_padded_venue_ids.cuda(), masked_padded_hour_ids.cuda())
                venue_output = venue_output.cpu()
                loss = self.criterion(venue_output.view(-1, self.venue_vocab_size),
                                      padded_venue_ids.flatten())
                total_loss += loss.item()

                # Get the predictions for the output venue IDs
                _, predicted_ids = torch.max(venue_output, dim=-1)

                print(predicted_ids[0], padded_venue_ids[0])

                ''''''
                # Get the first venue ID for each trajectory
                first_venue_target = padded_venue_ids[:, 0]
                first_venue_prediction = predicted_ids[:, 0]

                # Count the number of correctly predicted first venue IDs
                total_first_venues += first_venue_target.size(0)
                correct_predictions_first_venues += (first_venue_prediction == first_venue_target).sum().item()
                ''''''

                ''''''
                # Get the last venue ID for each trajectory (only support batch_size 1)
                last_venue_target = padded_venue_ids[:, -1]
                last_venue_prediction = predicted_ids[:, -1]

                # Count the number of correctly predicted last venue IDs (only support batch_size 1)
                total_last_venues += last_venue_target.size(0)
                correct_predictions_last_venues += (last_venue_prediction == last_venue_target).sum().item()
                ''''''

                # Flatten the predictions and ground truth for comparison
                predicted_ids = predicted_ids.view(-1)
                venue_target = padded_venue_ids.flatten()

                # Exclude the padded values from calculations
                non_padded_indices = venue_target != 0
                venue_target = venue_target[non_padded_indices]
                predicted_ids = predicted_ids[non_padded_indices]

                # Count the number of correctly predicted masked IDs
                total_ids += venue_target.size(0)  # length
                correct_predictions += (predicted_ids == venue_target).sum().item()

                # alter the predicted_ids
                alt_predicted_ids = torch.cat((venue_target[:1], predicted_ids[1:-1], venue_target[-1:]), dim=0)

                # Count the alter(max) f1\pairs_f1 and total f1\pairs_f1
                alt_f1 = f1_score(venue_target, alt_predicted_ids)
                alt_pairs_f1 = pairs_f1_score(venue_target, alt_predicted_ids)
                alt_f1_list.append(alt_f1)
                alt_pairs_f1_list.append(alt_pairs_f1)

                total_f1 = f1_score(venue_target, predicted_ids)
                total_pairs_f1 = pairs_f1_score(venue_target, predicted_ids)
                total_f1_list.append(total_f1)
                total_pairs_f1_list.append(total_pairs_f1)
                # Count the mean_dis and distor_dis
                # total_non_zero_ids = len([x for x in predicted_ids if x != 0])
                traj_dis_list = []

                for i in range(1, total_ids - 1):
                    if predicted_ids[i].item() == 0:
                        continue
                    venue_idx = venue_target[i].item()
                    predicted_idx = predicted_ids[i].item()
                    # convert the id to coordination
                    venue_coords = self.poi_dis_dict[venue_idx]
                    predicted_coords = self.poi_dis_dict[predicted_idx]
                    distance = calc_dist_vec(venue_coords[0], venue_coords[1], predicted_coords[0], predicted_coords[1])

                    traj_dis_list.append(distance)
                # judge the abnormal traj
                if len(traj_dis_list) == 0:
                    traj_dis = -1
                else:
                    traj_dis = np.mean(traj_dis_list)
                mean_dis_list.append(traj_dis)

                distor_dis_list = [calc_dist_vec(self.poi_dis_dict[predicted_ids[i].item()][0],
                                                 self.poi_dis_dict[predicted_ids[i].item()][1],
                                                 self.poi_dis_dict[predicted_ids[i - 1].item()][0],
                                                 self.poi_dis_dict[predicted_ids[i - 1].item()][1])
                                   for i in [1, -1]]

            epoch_loss = total_loss / len(self.eval_loader)
            '''eval precision'''
            precision = correct_predictions / total_ids
            print(f"eval precision: {precision}")

            '''first view precision'''
            first_venue_precision = correct_predictions_first_venues / total_first_venues
            print(f"eval first view precision: {first_venue_precision}")

            '''last view precision'''
            last_venue_precision = correct_predictions_last_venues / total_first_venues
            print(f"eval last view precision: {last_venue_precision}")

            if self.data_type == 'Flickr':
                # f1 and pairsf1
                '''alt_f1_score'''
                alt_f1_mean = np.mean(alt_f1_list)
                # print(f"max_f1_score: {max_f1_mean}")
                '''alt_pairs_f1_score'''
                alt_pairs_f1_mean = np.mean(alt_pairs_f1_list)
                # print(f"max_pairs_f1_score: {max_pairs_f1_mean}")
                '''total_f1_score'''
                total_f1_mean = np.mean(total_f1_list)
                # print(f"total_f1_score: {total_f1_mean}")
                '''total_pairs_f1_score'''
                total_pairs_f1_mean = np.mean(total_pairs_f1_list)
                # print(f"total_pairs_f1_score: {total_pairs_f1_mean}")
                '''distance'''
                mean_distance = np.mean(mean_dis_list)
                # print(f"mean_distance: {distance}")
                '''distortion'''
                distortion_distance = np.mean(distor_dis_list)
                # print(f"distortion_distance: {distortion_distance}")

                return epoch_loss, first_venue_precision, last_venue_precision, alt_f1_mean, alt_pairs_f1_mean, \
                    total_f1_mean, total_pairs_f1_mean, mean_distance, distortion_distance

            else:
                return epoch_loss, first_venue_precision, last_venue_precision
