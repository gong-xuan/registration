# registration
python mainvm.py --dataset hippocampus --bsize 64 --gpu 0 --num_workers 4 --weight 1,0.01,1 --feat --logfile hfeatb64 --epcoh 200 #94.8
python mainvm.py --dataset prostate --bsize 9 --gpu 1,2,3 --num_workers 3 --weight 1,0.01,1 --feat --logfile pfeatb9 #94.39
