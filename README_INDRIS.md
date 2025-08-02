./evaluate_grounding.py
- used to evaluate the grounding results using an assessor model (openai API), using the metadata files and the inference results from a model such as LLaVa-3D.
- Arguments:
  - `--infer`: path to the inference results from a model, such as LLaVa-3D
  - `--output`: path to save the processed results and scores
  - `--data`: path to the ScanRefer formatted data (e.g. ./data/scannet/scannet_grounding.json)
  - `--metadata`: path to the metadata file (e.g. ./data/scannet/metadata_scannet_grounding.json)
  - `--grounding`: optional, path to the processed grounding results for QA evaluation
- Before running:
  - Implement the `extract_pred` function to process the raw inference results from your model.
  - Set up your OpenAI API key.

./evaluate_qa.py
- used to evaluate the grounding and QA results respectively using an assessor model (openai API), using the metadata files and the inference results from a model such as LLaVa-3D.
- Arguments:
  - `--infer`: path to the inference results from a model, such as LLaVa-3D
  - `--output`: path to save the processed results and scores
  - `--data`: path to the ScanRefer or ScanQA formatted data (e.g. ./data/scannet/scannet_grounding.json or ./data/scannet/scannet_qa.json)
  - `--metadata`: path to the metadata file (e.g. ./data/scannet/metadata_scannet_grounding.json or ./data/scannet/metadata_scannet_qa.json)
  - `--grounding`: optional, path to the processed grounding results for QA evaluation
- Before running:
  - Implement the `extract_pred` function to process the raw inference results from your model.
  - Set up your OpenAI API key.

./utils.py
- utility functions used by the evaluation scripts, such as loading json files, processing results, cleaning assessee model (LLaVa-3D) responses and calculating scores.

./data/qa_to_scanqa_format.py
- generates ScanQA-formatted ScanNet QA annotations from the metadata. 

./data/grounding_to_scanrefer_format.py
- generates ScanRefer-formatted ScanNet grounding annotations from the metadata.

./data/system_prompt.json
- used by a large model, such as GPT via openai API, as a prompt; will judge on a scale of 1-5 the how good each prediction is (for example, the json passed through --infer to evaluate_qa.py that was generated from the model you're evaluating)

./data/scannet/metadata_scannet*
- the metadata, from which the annotation files ./data/scannet/scannet* (in associated formats) are generated
- keys in metadata_scannet_qa: dict_keys(['scene0025_00', 'scene0046_00', 'scene0050_00', 'scene0144_00', 'scene0164_00', 'scene0378_00', 'scene0426_00', 'scene0458_01', 'scene0518_00', 'scene0568_02', 'scene0591_02', 'scene0593_00', 'scene0616_00', 'scene0697_01', 'scene0699_00', 'scene0700_00'])

./data/scannet/scannet*
- the labels in ScanNet format, used by ScanRefer and ScanQA, generated from the metadata files using grounding_to_scanrefer_format.py and qa_to_scanqa_format.py respectively

./data/scannet/obj_id_to_class.json
- mapping (for convenience) from object id -> class name for ScanNet scenes, used by grounding_to_scanrefer_format.py and qa_to_scanqa_format.py






To use with LLaVa-3D:
data/scannet/scannet_qa.json is a list of dictionaries, each with the following (all the keys we need):
- question_id: scene0025_00_27_0_appearance_0
- scene_id: scene0025_00
- question: "What is the colour of the largest box?"
- answer: ['White'] (a list)


To use with KimiVL:
- Similar to above.
