python mainvm.py --debug --dataset prostate

#python mainvm.py --bsize 4 --gpu 0,1,2,3 --snapshot 1 --feat --logfile prosfeat_snapmopt0.1_c15e300 --epoch 300 --dataset prostate
# python mainvm.py --bsize 4 --gpu 0,1 --feat --dataset prostate --snapshot 1 --logfile pros_snapfeatmopt0.1_e300c15
# python mainvm.py --bsize 1 --gpu 3 --feat --dataset prostate --logfile debug
#python mainvm.py --bsize 128 --gpu 0,1,2 --uncert 1 --onlytrainvar --feat --logfile myvartr2
#python mainvm.py --bsize 96 --gpu 2,3 --uncert 1 --onlytrainvar --feat --logfile myvar --lr 1e-5 
#python mainvm.py --bsize 128 --gpu 0,1,2 --uncert 1 --onlytrainvar --logfile drop0.5-4
#python mainvm.py --bsize 96 --gpu 0,1 --bootstrap 0.9 --logfile bootfeat0.9-1 --feat
#python mainvm.py --bsize 96 --gpu 0,1 --droprate 0.1 --logfile drop0.1-3
#python mainvm.py --bsize 128 --gpu 0,1,2 --dataset prostate --logfile pros
#python mainvm.py --bsize 96 --gpu 0,1 --logfile tloss0 --weight '1,0,0.01' --debug
#python mainvm.py --bsize 1 --gpu 0,1,2,3 --dataset prostate --logfile pros --num_workers 0 --trpercent 0.01 --debug
#python mainvm.py --bsize 12 --gpu 0,1,2,3 --dataset prostate --logfile pros_feat --feat --epoch 200

git add . && git commit -m "initial commit"