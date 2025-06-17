#!/usr/bin/env python3
"""Check CoreML model specification."""

import coremltools as ct
import os

coreml_path = '/tmp/qwen-test/float32/test_qwen.mlpackage'
if os.path.exists(coreml_path):
    try:
        model = ct.models.MLModel(coreml_path)
        spec = model.get_spec()
        print('Model inputs:')
        for inp in spec.description.input:
            print(f'  {inp.name}: {inp.type}')
        print('\nModel outputs:')
        for out in spec.description.output:
            print(f'  {out.name}: {out.type}')
        print('\nModel states:')
        if hasattr(spec.description, 'stateTypes') and spec.description.stateTypes:
            for state in spec.description.stateTypes:
                print(f'  {state.name}: {state.type}')
        else:
            print('  No states found')
    except Exception as e:
        print(f'Error loading model: {e}')
else:
    print(f'Model not found at {coreml_path}') 