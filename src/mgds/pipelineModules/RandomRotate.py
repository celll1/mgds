import torch
from torchvision.transforms import functional, InterpolationMode

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import random

class RandomRotate(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: [str],
            enabled_in_name: str,
            fixed_enabled_in_name: str,
            max_angle_in_name: str,
    ):
        super(RandomRotate, self).__init__()
        self.names = names
        self.enabled_in_name = enabled_in_name
        self.fixed_enabled_in_name = fixed_enabled_in_name
        self.max_angle_in_name = max_angle_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.enabled_in_name] + [self.fixed_enabled_in_name] + [self.max_angle_in_name]

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        fixed_enabled = self._get_previous_item(variation, self.fixed_enabled_in_name, index)
        max_angle = self._get_previous_item(variation, self.max_angle_in_name, index)

        rand = self._get_rand(variation, index)
        item = {}

        if enabled:
            angle = rand.uniform(-max_angle, max_angle)
        elif fixed_enabled:
            angle = max_angle
        else:
            angle = 0.0

        for name in self.names:
            previous_item = self._get_previous_item(variation, name, index)
            if (enabled or fixed_enabled) and rand.uniform(0, 1) < 0.2:
                orig_dtype = previous_item.dtype
                img = previous_item.to(dtype=torch.float32) if orig_dtype == torch.bfloat16 else previous_item
                
                orig_h, orig_w = img.shape[1:3]
                
                rotated = functional.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
                
                rotated_h, rotated_w = rotated.shape[1:3]
                scale = min(orig_h / rotated_h, orig_w / rotated_w)
                
                new_h = int(rotated_h * scale)
                new_w = int(rotated_w * scale)
                rotated = functional.resize(rotated, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
                
                offset_y = int((new_h - orig_h) * rand.normalvariate(0, 0.25))
                offset_x = int((new_w - orig_w) * rand.normalvariate(0, 0.25))
                
                start_y = (new_h - orig_h) // 2 + offset_y
                start_x = (new_w - orig_w) // 2 + offset_x
                
                rotated = rotated[:, 
                                max(0, start_y):max(0, start_y) + orig_h,
                                max(0, start_x):max(0, start_x) + orig_w]
                
                item[name] = rotated.to(dtype=orig_dtype) if orig_dtype == torch.bfloat16 else rotated
            else:
                item[name] = previous_item

        return item
