import os
import pandas as pd

def files2csv(dataset_path):
	file_list = []
	labels = []

	for folder in os.listdir(dataset_path):

		folder_path = os.path.join(dataset_path, folder)
		for file in os.listdir(folder_path):

			file_path = folder_path+'/'+file

			file_list.append(file_path)
			labels.append(folder)

	columns = {"filename": file_list, "label": labels}

	data = pd.DataFrame(columns, index = None)
	return data


# inflam_detect
# train_dataset_path = "./data/dataset/inflam_detect/train/"
# val_dataset_path = './data/dataset/inflam_detect/val/'
# test_dataset_path = './data/dataset/inflam_detect/test/'
# csv_folder = './data/dataset/inflam_detect/csv/'

# inflam_classify
train_dataset_path = "./data/dataset/inflam_classify/train/"
val_dataset_path = './data/dataset/inflam_classify/val/'
test_dataset_path = './data/dataset/inflam_classify/test/'
csv_folder = './data/dataset/inflam_classify/csv/'


data = files2csv(train_dataset_path)
data.to_csv(os.path.join(csv_folder, 'train.csv'), index =None)

data = files2csv(val_dataset_path)
data.to_csv(os.path.join(csv_folder, 'val.csv'), index =None)

data = files2csv(test_dataset_path)
data.to_csv(os.path.join(csv_folder, 'test.csv'), index =None)