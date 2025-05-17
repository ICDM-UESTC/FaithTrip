import logging
import pandas as pd
import torch
import numpy as np


def load_data(dataset):
    """
    load data from csv file and then process it.
    (1) renumber the POI id, according to its frequency
    (2) renumber the user id, according to its frequency
    (3) number the category, according to its frequency
    (4) encode longitude and latitude (how?) -- current version: ignore
    -- maybe I can define multiple anchors, and calculate the distance between the current POI to them
    -- note: the longitude and latitude of a POI in (at) different trajectories (time) may vary
    -- note: it's ok, because we will concat the POI representation and spatial representation
    (5) encode UTC_time (how? need month, weekend / week days, day, hour?) -- current version: only hour, same to
    previous works
    (6) decompose data into multiple trajectories, each of which is a dict
    :param dataset:
    :return: grouped_df
    """
    path = f'./asset/data/{dataset}.csv'
    df = pd.read_csv(path)

    "(1) renumber the POI id, according to its frequency"
    poi_value_counts = df['venue_ID'].value_counts()
    poi_sorted_values = poi_value_counts.index.tolist()
    poi_renumber_dict = {value: i + 1 for i, value in enumerate(poi_sorted_values)}
    # print(poi_renumber_dict)
    df['venue_ID'] = df['venue_ID'].map(poi_renumber_dict)
    "(2) renumber the user id, according to its frequency"
    user_value_counts = df['user_ID'].value_counts()
    user_sorted_values = user_value_counts.index.tolist()
    user_renumber_dict = {value: i + 1 for i, value in enumerate(user_sorted_values)}
    # print(user_renumber_dict)
    df['user_ID'] = df['user_ID'].map(user_renumber_dict)
    "(3-1) number the category name, according to its frequency"
    cat_n_value_counts = df['venue_category_name'].value_counts()
    cat_n_sorted_values = cat_n_value_counts.index.tolist()
    cat_n_renumber_dict = {value: i + 1 for i, value in enumerate(cat_n_sorted_values)}
    # print(cat_n_renumber_dict)
    df['venue_category_name'] = df['venue_category_name'].map(cat_n_renumber_dict)
    "(3-2) number the category id, according to its frequency"
    cat_i_value_counts = df['venue_category_ID'].value_counts()
    cat_i_sorted_values = cat_i_value_counts.index.tolist()
    cat_i_renumber_dict = {value: i + 1 for i, value in enumerate(cat_i_sorted_values)}
    # print(cat_i_renumber_dict)
    df['venue_category_ID'] = df['venue_category_ID'].map(cat_i_renumber_dict)
    "(4)"

    "(5) encode UTC (1~24)"
    df['UTC_time'] = pd.to_datetime(df['UTC_time'])
    df['hour'] = df['UTC_time'].dt.hour
    "(6) merge the trajectory and time by traj_id"
    grouped_df = df.groupby('traj_id').agg({
        'user_ID': 'first',  # Keep the first user ID for each trajectory
        'venue_ID': list,  # Aggregate the POIs into a list for each trajectory
        'hour': list,  # Optional: Aggregate the UTC times into a list for each trajectory
        'latitude': list,
        'longitude': list
    }).reset_index()
    return grouped_df


def split_df_train_eval_test(df, random_seed, ratio):
    """
    split data into train_df, eval_df and test_df

    Parameters:
    :param df: The DataFrame to be split.
    :param random_seed: Seed for random number generator.
    :param ratio: The ratio in which to split the data.

    Returns:
    :return:Three DataFrames: train_df, eval_df, and test_df.
    """
    if not isinstance(ratio, tuple):
        raise TypeError("ratio must be a tuple.")
    if ratio[0] + ratio[1] + ratio[2] != 1:
        raise AssertionError('ratio should be normalized.')

    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # Calculate the sizes for train, eval, and test datasets
    total_size = len(df)
    train_size = int(ratio[0] * total_size)
    eval_size = int(ratio[1] * total_size)
    # Split the data into train, eval, and test datasets
    train_df = df[:train_size]
    eval_df = df[train_size:train_size + eval_size]
    test_df = df[train_size + eval_size:]
    return train_df, eval_df, test_df


def load_flickr_data(dataset):
    """
    load dataset and process like "load_data"
    :param dataset:
    :return: grouped_df, renumber_poi_distance_dict: {re_poi: (lon, lat)}
    """
    path = f'./asset/data/{dataset}.csv'
    df = pd.read_csv(path)

    "(1) renumber the POI id, according to its frequency"
    poi_value_counts = df['venue_ID'].value_counts()
    poi_sorted_values = poi_value_counts.index.tolist()
    poi_renumber_dict = {value: i + 1 for i, value in enumerate(poi_sorted_values)}
    # print(poi_renumber_dict)
    df['venue_ID'] = df['venue_ID'].map(poi_renumber_dict)

    "(2) renumber the user id, according to its frequency"
    user_value_counts = df['user_ID'].value_counts()
    user_sorted_values = user_value_counts.index.tolist()
    user_renumber_dict = {value: i + 1 for i, value in enumerate(user_sorted_values)}
    # print(user_renumber_dict)
    df['user_ID'] = df['user_ID'].map(user_renumber_dict)

    "(3) number the category name, according to its frequency"
    cat_n_value_counts = df['venue_category_name'].value_counts()
    cat_n_sorted_values = cat_n_value_counts.index.tolist()
    cat_n_renumber_dict = {value: i + 1 for i, value in enumerate(cat_n_sorted_values)}
    # print(cat_n_renumber_dict)
    df['venue_category_name'] = df['venue_category_name'].map(cat_n_renumber_dict)

    "(4) encode UTC (1~24)"
    df['UTC_time'] = pd.to_datetime(df['UTC_time'])
    df['hour'] = df['UTC_time'].dt.hour

    "(5) merge the trajectory and time by traj_id"
    grouped_df = df.groupby('traj_ID').agg({
        'user_ID': 'first',  # Keep the first user ID for each trajectory
        'venue_ID': list,  # Aggregate the POIs into a list for each trajectory
        'hour': list,  # Optional: Aggregate the UTC times into a list for each trajectory
        'latitude': list,  # each poi in certain trajectory
        'longitude': list  # each poi in certain trajectory
    }).reset_index()

    "(6) add renumber_poi_distance_dict to count the distance, also add padded number 0"
    renumber_poi_distance_df = df[['venue_ID', 'longitude', 'latitude']].drop_duplicates().sort_values(
        by='venue_ID').reset_index(drop=True)
    venue_data_zero = {'venue_ID': 0, 'longitude': 0.0, 'latitude': 0.0}
    venue_data_zero_df = pd.DataFrame([venue_data_zero])
    renumber_poi_distance_df = pd.concat([venue_data_zero_df, renumber_poi_distance_df], ignore_index=True)

    renumber_poi_distance_dict = renumber_poi_distance_df.groupby('venue_ID').apply(
        lambda x: (x['longitude'].iloc[0], x['latitude'].iloc[0])).to_dict()

    return grouped_df, renumber_poi_distance_dict


def split_df_leave_one_out(split_index, df, random_seed):
    """
    split data into train_df, test_df

    :param split_index:
    :param df:
    :param random_seed:
    :return:
    """
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # Calculate the sizes for train and test datasets

    train_df = df[df.index != split_index]
    test_df = df[df.index == split_index]

    return train_df, test_df


# Padding Function
def pad_sequences(sequences, pad_value=0, max_len=None):
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    if max_len:
        if padded_sequences.size(1) < max_len:
            padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max_len - padded_sequences.size(1)))
    return padded_sequences


def collate_fn(batch):
    masked_venue_ids, masked_hour_ids, venue_ids, masked_latitudes, masked_longitudes = zip(*batch)
    padded_venue_ids = pad_sequences(venue_ids, pad_value=0)
    masked_padded_latitudes = pad_sequences(masked_latitudes, pad_value=0)
    masked_padded_longitudes = pad_sequences(masked_longitudes, pad_value=0)
    masked_padded_venue_ids = pad_sequences(masked_venue_ids, pad_value=0)
    masked_padded_hours = pad_sequences(masked_hour_ids, pad_value=0)

    return masked_padded_venue_ids, masked_padded_hours, padded_venue_ids, masked_padded_latitudes, masked_padded_longitudes


def get_logger(filename, level=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[level])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    fh.setLevel(level_dict[level])
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # sh.setLevel(level_dict[verbosity])
    # logger.addHandler(sh)

    return logger


def f1_score(target, predict, noloop=False):
    """
    Compute F1 Score for recommended trajectories
    :param target: the actual trajectory
    :param predict: the predict trajectory
    :param noloop:

    :return: f1
    """
    assert (isinstance(noloop, bool))
    assert (len(target) > 0)
    assert (len(predict) > 0)

    if noloop:
        intersize = len(set(target) & set(predict))
    else:
        match_tags = np.zeros(len(target), dtype=np.bool_)
        for poi in predict:
            for j in range(len(target)):
                if not match_tags[j] and poi == target[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(target)
    precision = intersize * 1.0 / len(predict)
    denominator = recall + precision
    if denominator == 0:
        denominator = 1

    f1 = 2 * precision * recall * 1.0 / denominator

    return f1


def pairs_f1_score(target, predict):
    """
    Compute Pairs_F1 Score for recommended trajectories
    :param target:
    :param predict:
    :return: pairs_f1
    """
    # Check if number of elements > 0
    assert target.numel() > 0
    n = target.numel()
    nr = predict.numel()
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    order_dict = dict()
    for i, poi in enumerate(target):
        order_dict[poi.item()] = i

    nc = 0
    for i in range(nr):
        poi1 = predict[i].item()
        for j in range(i + 1, nr):
            poi2 = predict[j].item()
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        pairs_f1 = 0
    else:
        pairs_f1 = 2. * precision * recall / (precision + recall)

    return pairs_f1


def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin(np.sqrt(
        (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
    return dist


def count_adjacent_duplicates(numbers):
    if len(numbers) < 2:
        return 0  # If the list length is less than 2, there are no adjacent elements, and the number of duplicates is 0

    count = 0
    prev = numbers[0]  # Initialize the previous element as the first element in the list

    for current in numbers[1:]:  # Start traversing from the second element of the list
        if current == prev:
            count += 1  # If the current element is the same as the previous element, increase the repeat count
        prev = current  # Update the previous element to the current element

    return count

def count_best_index(lst1, lst2, lst3):
    best_indices = [i for i, value in enumerate(lst1) if value ==
                    max(lst1)]
    sublist_2 = [lst2[index] for index in best_indices]

    best_indices = [value for i, value in enumerate(best_indices) if sublist_2[i] ==
                    max(sublist_2)]

    sublist_3 = [lst3[index] for index in best_indices]
    best_indices = [value for i, value in enumerate(best_indices) if sublist_3[i] ==
                    min(sublist_3)]
    best_index = best_indices[-1]

    return best_index

