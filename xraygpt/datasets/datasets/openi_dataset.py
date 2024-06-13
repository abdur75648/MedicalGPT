"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from PIL import Image
import webdataset as wds
from xraygpt.datasets.datasets.base_dataset import BaseDataset
from xraygpt.datasets.datasets.caption_datasets import CaptionDataset


# class OpenIDataset(BaseDataset):
#     def __init__(self, vis_processor, text_processor, location):
#         super().__init__(vis_processor=vis_processor, text_processor=text_processor)

#         self.inner_dataset = wds.DataPipeline(
#             wds.ResampledShards(location),
#             wds.tarfile_to_samples(handler=wds.warn_and_continue),
#             wds.shuffle(1000, handler=wds.warn_and_continue),
#             wds.decode("pilrgb", handler=wds.warn_and_continue),
#             wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
#             wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
#             wds.map(self.to_dict, handler=wds.warn_and_continue),
#         )

#     def to_dict(self, sample):
#         return {
#             "image": sample[0],
#             "text_input": self.text_processor(sample[1]["caption"]),
#         }
    
class OpenIDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        # img_file = '{}.png'.format(ann["image_id"])
        # image_path = os.path.join(self.vis_root, img_file)
        image_path = ann["image"]
        image_path = os.path.join("dataset", image_path)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann['caption']
        prompt = ann['prompt']

        return {
            "image": image,
            "caption":caption,
            "prompt":prompt,    
            "image_id": self.img_ids[ann["image"]],
        }

