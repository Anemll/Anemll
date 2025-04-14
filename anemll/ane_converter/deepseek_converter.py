#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

#use LLamaModel for DeepSeek on LLaMA 3.1 architecture

'''
from .base_converter import BaseConverter
import coremltools as ct

class DeepSeekConverter(BaseConverter):
    """Handles DeepSeek model conversion to Apple Neural Engine format."""

    def convert(self):
        self.preprocess()
        # DeepSeek model needs special transformations before ANE conversion
        ane_model = self.convert_to_ane(self.model)
        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = ane_model  # Set for postprocess
            self.postprocess(num_workers=None)  # Allow passing num_workers if needed
            ane_model = self.converted_model
        return ane_model

    def convert_to_ane(self, model):
        """Convert DeepSeek model to Apple Neural Engine format using CoreMLTools."""
        return ct.convert(
            model,
            compute_units=ct.ComputeUnit.ALL,  # Enables Apple Neural Engine
            minimum_deployment_target=ct.target.iOS16  # Required for ANE support
        )
'''