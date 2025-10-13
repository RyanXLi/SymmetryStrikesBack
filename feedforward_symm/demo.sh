zsnew_ckpt_path="/replace/with/your/own/path/latest.ckpt"
zsnew_yaml_path="./options/shape.onlysymm.4a100.freeze.yaml"
# replace the following path with your own
input_path="/replace/with/your/own/path/datasets/data/gso/2_of_Jenga_Classic_Game_225_15.png"
filename=$(basename "${input_path}")
filename_no_ext="${filename%.*}"
dataset_name="gso"
output_path="./output/${dataset_name}/${filename_no_ext}"

python demo.py --yaml="${zsnew_yaml_path}" --input_image_path="${input_path}" --output_dir="${output_path}" --task=shape --ckpt="${zsnew_ckpt_path}"