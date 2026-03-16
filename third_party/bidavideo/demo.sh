#!/bin/bash

set -ex

source /opt/conda/bin/activate ""
conda activate bidavideo

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# # RAFTStereo + BiDAStabilizer
# python demo.py --model_name raftstereo --stabilizer \
#  --ckpt ./checkpoints/raftstereo_robust/raftstereo_robust.pth \
#  --stabilizer_ckpt ./checkpoints/raftstereo_stabilizer_robust/raftstereo_stabilizer_robust.pth \
#  --path ./demo_video/ --output_path ./demo_output/RAFTStereo_BiDAStabilizer/ --save_png

#  # IGEVStereo + BiDAStabilizer
# python demo.py --model_name igevstereo --stabilizer \
#  --ckpt ./checkpoints/igevstereo_robust/igevstereo_robust.pth \
#  --stabilizer_ckpt ./checkpoints/igevstereo_stabilizer_robust/igevstereo_stabilizer_robust.pth \
#  --path ./demo_video/ --output_path ./demo_output/IGEVStereo_BiDAStabilizer/ --save_png

# # BiDAStereo
# python demo.py --model_name bidastereo --ckpt ./checkpoints/bidastereo_robust/bidastereo_robust.pth \
#          --path ./demo_video/ --output_path ./demo_output/BiDAStereo/ --save_png

# # RAFTStereo
# python demo.py --model_name raftstereo --ckpt ./checkpoints/raftstereo_robust/raftstereo_robust.pth \
#          --path ./demo_video/ --output_path ./demo_output/RAFTStereo/ --save_png

# # IGEVStereo
# python demo.py --model_name igevstereo --ckpt ./checkpoints/igevstereo_robust/igevstereo_robust.pth \
#          --path ./demo_video/ --output_path ./demo_output/IGEVStereo/ --save_png




# BiDAStereo
# python demo.py --model_name bidastereo --ckpt checkpoints/bidastereo_robust/bidastereo_robust.pth \
#          --path /home/jupyter/datasets/tartanair/unzipped/abandonedfactory/Easy/P010 --output_path ./outputs2/BiDAStereo/ --save_png \
#          --resize 480 640 

# python demo.py --model_name bidastereo --ckpt checkpoints/bidastereo_robust/bidastereo_robust.pth \
#          --path ./demo_video/ --output_path ./outputs2/BiDAStereo/ --save_png


python demo.py --model_name bidastereo --ckpt checkpoints/bidastereo_robust/bidastereo_robust.pth \
         --path ./demo_video/ --output_path ./outputs2/BiDAStereo_swaped/ --save_png


# # RAFTStereo
# python demo.py --model_name raftstereo --ckpt ./checkpoints/raftstereo_robust/raftstereo_robust.pth \
#          --path ./demo_video/ --output_path ./outputs2/RAFTStereo/ --save_png

# # IGEVStereo
# python demo.py --model_name igevstereo --ckpt ./checkpoints/igevstereo_robust/igevstereo_robust.pth \
#          --path ./demo_video/ --output_path ./outputs2/IGEVStereo/ --save_png


         
