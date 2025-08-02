./system_prompt.json
- Used by a large model, such as GPT via openai API, as a prompt; will judge on a scale of 1-5 the how good each prediction is (for example, the json passed through --infer to evaluate_qa.py that was generated from the model you're evaluating)

./grounding_to_scanrefer_format.py
- used to generate ./scannet/scannet_grounding.json from ./metadata_scannet_grounding.json

./qa_to_scanqa_format.py
- used to generate ./scannet/scannet_qa.json from ./metadata_scannet_qa.json
