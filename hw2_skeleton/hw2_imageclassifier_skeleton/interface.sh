#
DEVICE=cuda
EPOCHS=50


#
cnn=0
cgcnn=0
shuffle=0
amp=0
for arg in $*; do
    #
    elif [[ ${arg} == "cnn" ]]; then
        #
        cnn=1
    elif [[ ${arg} == "cgcnn" ]]; then
        #
        cgcnn=1
    elif [[ ${arg} == "shuffle" ]]; then
        #
        shuffle=1
    elif [[ ${arg} == "amp" ]]; then
        #
        amp=1
    else
        #
        echo "unknown interface ${arg}."
        exit 1
    fi
done

EPOCHS=100
#
if [[ ${cnn} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    sbatch scholar.sh python main.py --batch-size 100 --cnn --lr 1e-3 \
        --kernel 5 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS}
    sbatch scholar.sh python main.py --batch-size 100 --cnn --lr 1e-3 \
        --kernel 3 --stride 3 --device ${DEVICE} --num-epochs ${EPOCHS}
    sbatch scholar.sh python main.py --batch-size 100 --cnn --lr 1e-3 \
        --kernel 14 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS}
fi

#
if [[ ${shuffle} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    sbatch scholar.sh python main.py --batch-size 100 --cgcnn --lr 1e-2 \
        --shuffle-label --device ${DEVICE} --num-epochs ${EPOCHS}
fi


EPOCHS=50
#
if [[ ${cgcnn} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    sbatch scholar.sh python main.py --batch-size 100 --cgcnn --lr 1e-3 \
        --kernel 5 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS} --rot-flip
fi

