class PawDataset(Dataset):
    def __init__(self, data_source, metadata, H = 128, W = 128, test_data = False):
        super(PawDataset, self).__init__()
        self.data_source = data_source
        self.metadata = metadata
        self.H = H
        self.W = W
        self.test_data = test_data
        self.augment = self.transform()

    def transform(self):
        augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
            ]
        )
        return augmentation

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # image_link
        source = self.metadata['Id'][index]
        source = os.path.join(f"{self.data_source}{source}.jpg")
        # loading the image and tranforming it into a torh tensor
        image = self.load_image(source)
        # loading metadata
        metadata = self.metadata.iloc[index, 1:13].astype('float32').to_numpy().reshape(1,-1)
        if self.test_data == False:
            # target output
            image = self.augment(image)
            image = transforms.ToTensor()(image)
            target = self.metadata['Pawpularity'][index] / 100.0
            return (image, metadata, target)
        else:
            image = transforms.ToTensor()(image)
            return (image, metadata)
    def load_image(self, source):
        img = cv2.imread(source)
        img = cv2.resize(img, (self.H, self.W))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img