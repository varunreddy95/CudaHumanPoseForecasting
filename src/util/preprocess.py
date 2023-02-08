from ast import main
from typing import List
from paths import DATASET_DIR, H36_DIR
import os
import pickle
from tqdm import tqdm


def keep_every_n_frame_of_sequence(data, n, filename):
    sequence_index = data[0]['video_id']
    count = 0
    processed_data = list()

    for i in tqdm(range(len(data))):
        if sequence_index != data[i]['video_id']:
            sequence_index = data[i]['video_id']
            count = 0
        if count % n == 0:
            processed_data.append(data[i])

        count = count + 1

    return processed_data


def make_nested_list_from_data(data: List):
    sequence_list = list()
    processed_data = list()
    sequence_index = data[0]['video_id']

    for i in tqdm(range(len(data))):
        if sequence_index != data[i]['video_id']:
            sequence_index = data[i]['video_id']
            processed_data.append(sequence_list)
            sequence_list = list()

        sequence_list.append(data[i])

    processed_data.append(sequence_list)

    return processed_data

def make_combinations_from_nested_list(data: List):
    processed_data = list()
    for idx, value in enumerate(data):

        for i in range (0, len(value), 20):
            temporary = list()
            if i+20 < len(value):
                for x in range(i, i+20):
                    temporary.append(value[x])
                processed_data.append(temporary)

        start = len(value)-20
        temporary = list()
        for i in range(start, start+20):
            temporary.append(value[i])

        processed_data.append(temporary)

    print(len(temporary))

    print(len(processed_data))
    print(len(processed_data[0]))
    return processed_data






def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # with open(os.path.join(H36_DIR, "h36m_validation.pkl"), "rb") as file:
    #     data = pickle.load(file)

    # data = keep_every_n_frame_of_sequence(data, 8, os.path.join(
    #     H36_DIR, "h36m_validation_processed.pkl"))

    # data = make_nested_list_from_data(data)
    # save_data_as_pickle(data, os.path.join(
    #     H36_DIR, "h36m_validation_processed.pkl"))

    # with open(os.path.join(H36_DIR, "h36m_train.pkl"), "rb") as file:
    #     data = pickle.load(file)

    # data = keep_every_n_frame_of_sequence(data, 8, os.path.join(
    #     H36_DIR, "h36m_train_processed.pkl"))
    # data = make_nested_list_from_data(data)

    # save_data_as_pickle(data, os.path.join(
    #     H36_DIR, "h36m_train_processed.pkl"))

    with open(os.path.join(H36_DIR, "h36m_train.pkl"), "rb") as file:
        data = pickle.load(file)

    data = keep_every_n_frame_of_sequence(data, 8, os.path.join(
        H36_DIR, "h36m_train_processed.pkl"))
    data = make_nested_list_from_data(data)

    data = make_combinations_from_nested_list(data)


    save_data_as_pickle(data, os.path.join(
        H36_DIR, "h36m_train_processed_all_sequence.pkl"))

    with open(os.path.join(H36_DIR, "h36m_validation.pkl"), "rb") as file:
        data = pickle.load(file)

    data = keep_every_n_frame_of_sequence(data, 8, os.path.join(
        H36_DIR, "h36m_validation_processed.pkl"))

    data = make_nested_list_from_data(data)
    data = make_combinations_from_nested_list(data)
    save_data_as_pickle(data, os.path.join(
        H36_DIR, "h36m_validation_processed_all_sequence.pkl"))