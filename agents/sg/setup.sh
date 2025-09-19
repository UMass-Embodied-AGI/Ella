cd builder/builtin
make
cd ../..

piplist=$(pip list)

if [[ $piplist != *"efficient-sam"* ]]; then
    echo "installing efficient-sam"
    cd third_party/EfficientSAM
    pip install -e .
    cd ../..
fi

if [[ $piplist != *"recognize-anything"* ]]; then
    echo "installing recognize-anything"
    cd third_party/recognize-anything
    pip install -e .
    cd ../..
fi

if [[ $piplist != *"groundingdino"* ]]; then
    echo "installing groundingdino"
    # if on unity server, uncomment the following line
#    module load gcc/9.4.0
#    export CUDA_HOME=/modules/opt/linux-ubuntu24.04-x86_64/nvhpc/Linux_x86_64/24.9/cuda/12.6
    cd third_party/GroundingDINO
    pip install -e .
    cd ../..

    # fix opencv-python conflict by supervision
    pip uninstall opencv-python-headless -y
    pip uninstall opencv-python -y
    pip install opencv-python==4.9.0.80
fi

if [[ $piplist != *"open_clip_torch"* ]]; then
    echo "installing open_clip_torch"
    cd third_party/open_clip
    pip install -e .
    cd ../..
fi

# if [[ $piplist != *"SAM-2"* ]]; then
#     echo "installing SAM-2"
#     cd third_party/segment-anything-2
#     pip install -e .
#     cd ../..
# fi