import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from gauss_diffusion_process import GaussianDiffusion, Sampler
from diffusion_model import TransformerLayer
from rich.progress import track


class TrajDiffusionTrainer:
    def __init__(self,
                 train_dataset,
                 eval_dataset,
                 poi_dis_dict,
                 data_type,
                 wandb_or_not,
                 step,
                 max_noise,
                 lr,
                 batch_size,
                 num_epochs,
                 d_model,
                 num_encoder_layers,
                 venue_vocab_size,
                 hour_vocab_size,
                 max_length_venue_id,
                 fast_sampling):

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.poi_dis_dict = poi_dis_dict
        self.wandb_or_not = wandb_or_not
        self.step = step
        self.max_noise = max_noise
        self.data_type = data_type
        self.num_epochs = num_epochs
        self.venue_vocab_size = venue_vocab_size
        self.hour_vocab_size = hour_vocab_size

        # build dataloader
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       collate_fn=collate_fn,
                                       shuffle=True,
                                       drop_last=False)

        self.eval_loader = DataLoader(eval_dataset,
                                      batch_size=1,
                                      collate_fn=collate_fn,
                                      drop_last=False)

        # build_model
        self.model = GaussianDiffusion(beta_1=1e-4,
                                       beta_T=self.max_noise,
                                       T=self.step,  # 32
                                       theta=0.5,  # 0.5
                                       venue_vocab_size=self.venue_vocab_size,
                                       hour_vocab_size=self.hour_vocab_size,
                                       max_length_venue_id=max_length_venue_id,
                                       n_head=4,
                                       num_encoder_layers=num_encoder_layers,
                                       d_model=d_model).cuda()
        # build sampler
        self.sampler = Sampler(diffusion_model=self.model.diffusion_model,
                               venue_embedding=self.model.venue_embedding,
                               hour_embedding=self.model.hour_embedding,
                               beta_1=1e-4,
                               beta_T=self.max_noise,
                               T=self.step,
                               n_0=1,
                               gamma=0.1,
                               lambdas=25,
                               fast_sampling = fast_sampling,
                               mean_type='xstart',
                               var_type='fixedlarge').cuda()

        # optimizer setting
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        epoch_start_list = []
        epoch_end_list = []
        epoch_f1_score_list = []
        epoch_pairs_f1_list = []
        epoch_distance_list = []
        epoch_distortion_distance_list = []

        for epoch in tqdm(range(1, self.num_epochs + 1)):

            self.model.train()
            print(f'start the {epoch} epoch training...')

            total_loss = 0.0
            # load batch
            for masked_padded_venue_ids, masked_padded_hour_ids, padded_venue_ids, padded_hour_ids, _, _ in \
                    self.train_loader:

                self.optimizer.zero_grad()

                _, loss = self.model(padded_venue_ids.cuda(), masked_padded_venue_ids.cuda(),
                                     padded_hour_ids.cuda(), masked_padded_hour_ids.cuda())
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / len(self.train_loader)
            print(f'the {epoch} epoch train loss: {epoch_loss}')

            if self.wandb_or_not:
                wandb.log({'epoch': epoch, 'loss': epoch_loss})

            print(f"the {epoch} epoch training is over...start evaluate")
            # evaluate
            if self.data_type == "Flickr":
                eval_loss, start_accuracy, end_accuracy, f1, pairs_f1, distance, distortion_distance = self.evaluate()
                epoch_start_list.append(start_accuracy)
                epoch_end_list.append(end_accuracy)
                epoch_f1_score_list.append(f1)
                epoch_pairs_f1_list.append(pairs_f1)
                epoch_distance_list.append(distance)
                epoch_distortion_distance_list.append(distortion_distance)

            else:
                if epoch % 10 == 0:
                    eval_loss, start_ratio, end_ratio = self.evaluate()
                    print(eval_loss)

        # calculate results
        f1_max_index = count_best_index(epoch_f1_score_list, epoch_pairs_f1_list, epoch_distance_list)
        # pick the max f1 in each trajectory
        return (epoch_start_list[f1_max_index], epoch_end_list[f1_max_index], epoch_f1_score_list[f1_max_index],
                epoch_pairs_f1_list[f1_max_index], epoch_distance_list[f1_max_index],
                epoch_distortion_distance_list[f1_max_index])

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
        f1_list = []
        pairs_f1_list = []

        # for the mean_dis and distor_dis
        mean_dis_list = []
        distor_dis_list = []

        with torch.no_grad():
            for masked_padded_venue_ids, masked_padded_hour_ids, padded_venue_ids, padded_hour_ids, _, _ in \
                    self.eval_loader:

                print(f'start the evaluation...')

                predicted_ids = self.sampler(masked_padded_venue_ids.cuda(), masked_padded_hour_ids.cuda())

                # Get the predictions for the output venue IDs
                # _, predicted_ids = torch.max(predicted_ids, dim=-1)
                predicted_ids = predicted_ids.cpu()

                print("recommend:", predicted_ids[0], "target:", padded_venue_ids[0])

                # with open("glas.txt", "a") as file:
                #     file.write(f"====================================== + \n")
                #     file.write(f"recommend: {predicted_ids[0]}, target: {padded_venue_ids[0]}")


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
                non_padded_indices = (venue_target != 0)
                venue_target = venue_target[non_padded_indices]
                predicted_ids = predicted_ids[non_padded_indices]

                # Count the number of correctly predicted masked IDs
                total_ids += venue_target.size(0)
                correct_predictions += (predicted_ids == venue_target).sum().item()

                # Count the f1_score and pairs_f1_score
                f1 = f1_score(venue_target, predicted_ids)
                pairs_f1 = pairs_f1_score(venue_target, predicted_ids)
                f1_list.append(f1)
                pairs_f1_list.append(pairs_f1)
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
            precision = correct_predictions / total_ids
            # print(f"eval precision: {precision}")

            ''''''
            first_venue_precision = correct_predictions_first_venues / total_first_venues
            # print(f"eval first view precision: {first_venue_precision}")
            ''''''

            ''''''
            last_venue_precision = correct_predictions_last_venues / total_first_venues
            # print(f"eval last view precision: {last_venue_precision}")
            ''''''

            if self.data_type == 'Flickr':
                '''f1_score'''
                f1_mean = np.mean(f1_list)
                # print(f"f1_score: {f1_mean}")
                '''pairs_f1_score'''
                pairs_f1_mean = np.mean(pairs_f1_list)
                # print(f"pairs_f1_score: {pairs_f1_mean}")
                '''distance'''
                mean_distance = np.mean(mean_dis_list)
                # print(f"mean_distance: {distance}")
                '''distortion'''
                distortion_distance = np.mean(distor_dis_list)
                # print(f"distortion_distance: {distortion_distance}")

                return epoch_loss, first_venue_precision, last_venue_precision, f1_mean, pairs_f1_mean, mean_distance, \
                       distortion_distance
            else:
                return epoch_loss, first_venue_precision, last_venue_precision
