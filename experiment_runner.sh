#!/bin/bash

# ensure that OPENAI_API_KEY is set in the environment!


# just an initial run to see if evaluate_qa will start up; note that this uses SQA3D not Beacon so it does not run to completion.
python evaluate_qa.py --infer /data/SceneUnderstanding/SU_cursor/LLaVA-3D/experiments/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_pred-answers.json --output scrap.json
