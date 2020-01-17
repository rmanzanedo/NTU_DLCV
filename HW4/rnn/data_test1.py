import reader
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        # self.mode = mode
        # self.data_dir = args.data_dir
        self.vid_dir = args.valid
        # self.label_dir= os.path.join(self.data_dir, 'label')


        ''' read the data list '''
        self.label_path = os.path.join(args.gt)
        self.data = reader.getVideoList(self.label_path)

        ''' set up image path '''
        # print(self.data['Video_category'], len(self.data))
        # exit()

        ''' set up image trainsform '''


    def __len__(self):
        return len(self.data['Video_name'])

    def __getitem__(self, idx):
        # print(idx)
        ''' get data '''
        frames = reader.readShortVideo(self.vid_dir, self.data['Video_category'][idx],self.data['Video_name'][idx])
        # print(frames.shape)
        # frames = Image.fromarray(frames)

        label = int(self.data['Action_labels'][idx])
        ''' read image '''
        # print(idx)

        # return torch.from_numpy(frames),label
        return frames ,label