scHSCpath='/home/zhangjingxiao/.conda/envs/scHSC/bin/python'
Datasets=('Adam' 'Bach' 'Heart_lv' 'Klein' 'Muraro' 'Macosko' 'Plasschaert' 'Pollen' 'Quake_10x_Bladder' 'Quake_10x_Limb_Muscle' 
       'Quake_10x_Spleen' 'Quake_Smart-seq2_Diaphragm' 'Quake_Smart-seq2_Heart' 'Quake_Smart-seq2_Limb_Muscle'
       'Quake_Smart-seq2_Trachea' 'Romanov' 'Tosches_turtle' 'Young')

for dataset in ${Datasets[@]};
do
    nohup ${scHSCpath} run_scHSC.py --dataset $dataset  >/dev/null  2>&1
done
