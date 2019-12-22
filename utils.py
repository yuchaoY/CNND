from scipy import io
import numpy as np

def bbox_mask_generator(inner, outer):

    mask = np.ones([outer]*2)
    zero_index = int((outer-inner)/2)
    mask[zero_index:-zero_index, zero_index:-zero_index] = 0
    mask = mask.astype(bool)
    return mask

# loading testing data with mat format
def load_standard_mat(mat_file, gt=False):

    dataset = io.loadmat(mat_file)
    data = dataset['data'].astype(np.float32)
    if gt:
        labels = dataset['gt'].astype(np.float32)
    else:
        labels = dataset['groundtruth'].astype(np.float32)
    print('data shape:', data.shape)
    print('label shape:', labels.shape)
    return data, labels

# loading training data with mat format
# Download from http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
def load_salinas_mat(data_path, gt_path, remove_bands=True):
    data = io.loadmat(data_path)['salinas'].astype(np.float32)
    labels = io.loadmat(gt_path)['salinas_gt']

    # Delete several bands for keeping the the same bands with testing dataset (from 224 decrease to 189)
    if remove_bands:
        bands = np.concatenate((np.arange(6, 32),
                                   np.arange(35, 96),
                                   np.arange(97, 106),
                                   np.arange(113, 152),
                                   np.arange(166, 220)))
        data = data[:, :, bands]

    print('data shape:', data.shape)
    print('label shape:', labels.shape)
    return data, labels

# generate training data
def training_data_generator(data_path, gt_path, sim_samples=100, dis_samples=600, remove_bands=True):

    img, gt = load_salinas_mat(data_path, gt_path, remove_bands=remove_bands) # img:(H, W, d), gt:(H, W)

    d = img.shape[2]
    class_num = len(np.unique(gt))-1

    img = img.astype(np.float32)
    gt = gt.astype(np.float32)

    img = img.reshape(-1, d)
    gt = gt.flatten()
    class_num = len(np.unique(gt))-1 # except anomaly pixel

    training_data = np.zeros((1, d))
    training_labels = np.zeros(1)

    # dissimilar pixel pair
    for i in range(class_num-1):
        for j in range(i+1, class_num):

            index_i = np.where(gt == i+1)
            index_j = np.where(gt == j+1)
            data_i = img[index_i][:dis_samples]
            data_j = img[index_j][:dis_samples]

            training_data = np.concatenate((training_data, np.abs(data_i-data_j)), axis=0)

    training_labels = np.concatenate((training_labels, np.ones(int(dis_samples*class_num*(class_num-1)/2))))

    # similar pixel pair
    for i in range(class_num):
        data_similar = np.zeros((sim_samples, sim_samples, d))
        index = np.where(gt == i + 1)
        data = img[index][:sim_samples]
        for j in range(sim_samples):
            data_similar[j] = np.abs(data - data[j, None])
        data_similar = data_similar[np.triu_indices(sim_samples, k = 1)]
        training_data = np.concatenate((training_data, data_similar), axis=0)

    training_labels = np.concatenate((training_labels, np.zeros(int(class_num*sim_samples*(sim_samples-1)/2))))
    training_data = np.delete(training_data, 0, 0)
    training_labels = np.delete(training_labels, 0).astype(np.float32)
    training_data = training_data.reshape(training_data.shape[0], 1, training_data.shape[1]).astype(np.float32)
    return training_data, training_labels

# generate testing data
def data_generator_mask(img, inner, outer):

    img = img.astype(np.float32)

    new_data = []
    mask = bbox_mask_generator(inner, outer)
    radius = int((outer-1)/2)

    img = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), constant_values=-1)
    H = img.shape[0]
    W = img.shape[1]

    for i in range(radius, H-radius):
        for j in range(radius, W-radius):
            pixel = img[i, j, None]
            bbox = img[i-radius:i+radius+1, j-radius:j+radius+1]
            bbox = bbox[mask]
            bbox = bbox[bbox[:, 0] > 0]

            new_data.append(np.abs(bbox - pixel))
    return new_data


