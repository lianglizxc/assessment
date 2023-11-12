from transformers import EfficientNetForImageClassification, EfficientNetConfig
from torchvision import transforms as T
from PIL import Image
import torch
import os


class Inference():

    def __init__(self, model_config):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = EfficientNetConfig.from_json_file(model_config)
        self.model = EfficientNetForImageClassification(config)
        self.model = self.model.to(self.device)
        self.transform = T.Compose(
            [
                T.Resize([128, 128]),
                T.ToTensor(),
                T.Normalize(mean=[0.43, 0.45, 0.45], std=[0.229, 0.224, 0.225])
            ]
        )

    def load_latest_checkpoint(self, model_dir):
        maxtime = None
        lastest_file = None
        for file_name in os.listdir(model_dir):
            if file_name.endswith('ckpt'):
                file_name = os.path.join(model_dir, file_name)
                file_time = os.path.getctime(file_name)
                if maxtime is None:
                    maxtime = file_time
                    lastest_file = file_name
                elif file_time > maxtime:
                    maxtime = file_time
                    lastest_file = file_name
        print(f"found latest checkpoint {lastest_file.split('/')[-1]}")
        checkpoint = torch.load(lastest_file, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('load checkpoint completed')

    def prediction(self, image_path):
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        image = torch.unsqueeze(image, dim=0)
        logits = self.model(image).logits
        pred = torch.argmax(logits, -1)
        return pred.numpy().tolist()