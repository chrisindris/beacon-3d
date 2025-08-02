./metadata_scannet_qa.json:
- has the annotations for QA for ScanNet alongside grounding text (grounding text from ./metadata_scannet_grounding.json), organized by ScanNet scene and object within that scene (since beacon is object-centric)
- Each scene is a dictionary where the keys are a class of object i.e. "box" and the values are themselves keys that are numeric (object ids). Obj_id_to_class.json can be calculated from the metadata, but is a convenient mapping from object id to class name for the scene.  

./metadata_scannet_grounding.json:
- has the grounding annotations for ScanNet, organized by ScanNet scene and object within that scene
- 'extra knowledge' seems to be when the question needs more than just the objects mentioned in the grounding text to answer it.

./obj_id_to_class.json:
- for each scene, has the mapping from object id to class name for ../grounding_to_scanrefer_format.py and ../qa_to_scanqa_format.py, for generating ../scannet/scannet_grounding.json and ../scannet/scannet_qa.json using ../metadata_scannet_grounding.json and ../metadata_scannet_qa.json respectively

./scannet_qa.json:
- has the ScanNet QA data in the format used by ScanQA, with grounding text from ./scannet_grounding.json. This is the metadata_scannet_qa once converted to ScanQA format using ../qa_to_scanqa_format.py.
- each item is a question, with question_id like scene0025_00_27_0_appearance_0:
    - scene0025_00: the scene id
    - 27: the object id
    - 0: the question number for this object
    - appearance: the tag, or KNOWLEDGE_TYPE for this question (e.g. appearance, spatial, etc.)
    - 0: if extra knowledge is needed to answer the question, this will be 1, otherwise it will be 0. 

./scannet_grounding.json:
- has the ScanNet grounding data in the format used by ScanRefer, with grounding text from ./metadata_scannet_grounding.json. This is the metadata_scannet_grounding once converted to ScanRefer format using ../grounding_to_scanrefer_format.py.
- Each item is a link in a chain, labelled like: scene0025_00_27_chain_1_0_class-appearance-spatial, for which the terms are:
    - scene0025_00: the scene id
    - 27: the object id
    - chain_1: the chain number (for this object)
    - 0: the link number in the chain; as the link number increases, the link becomes more simple, less specific (a chain-of-thought) 
    - class-appearance-spatial: indicates the KNOWLEDGE_TYPE(s) for this link.  
