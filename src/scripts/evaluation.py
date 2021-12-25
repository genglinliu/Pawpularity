import pandas as pd

import torch
import torch.nn as nn

from scripts.utils import get_dataframe


def evaluate(test_loader, data_dir, model, device):
    model.eval() 
    print('Making predictions...')
    test_pred = []    
    test_df = get_dataframe(data_dir, is_train=False)
    
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device).float()
            # forward
            out = model(test_images)
            test_pred += out.cpu().numpy().tolist()

        # write to file
        output = pd.DataFrame({"Id": test_df['Id'], "Pawpularity": test_pred})
        output.to_csv('submission.csv', index = False)

        # check output
        output_df = pd.read_csv('submission.csv')

        return output_df
