#!/bin/bash
dataset='minerl'
batch=8
is=36
os=50
a=25
b=40
framedims=64
sample_size=$((is + os))
jobnum=1

models='VidODE VidODENRU VidODENRU2'
epochs=(1000)
nlayers=(2 3 4)
convcells=(1 2 3)

 > out/job_logs.txt
echo "Train ID      Test ID      Output File      Log Folder\n" > out/job_logs.txt

for epoch in ${epochs[@]}
do 
    ckpt=$((a * epoch))
    img=$((b * epoch))
    train_time=""
    test_time=""

    if [ $epoch -le 100 ]
    then
        train_time="12:00:00"
        test_time="0:30:00"

    elif [ $epoch -le 200 ]
    then
        train_time="24:00:00"
        test_time="0:30:00"

    elif [ $epoch -le 500 ]
    then
        train_time="24:00:00"
        test_time="0:30:00"

    elif [ $epoch -le 1000 ]
    then
        train_time="52:00:00"
        test_time="1:00:00"
    fi

    for model in ${models}
    do
        for nlayer in ${nlayers[@]}
        do
            for convcell in ${convcells[@]}
            do
                nru='False'
                nru2='False'
                train_output_file="out/""$model""/train/""$model""_e""$epoch""_""$nlayer""l_""$convcell""c_""$batch""b_""$framedims""f_TRAIN-%j.out" 
                test_output_file="out/""$model""/test/""$model""_e""$epoch""_""$nlayer""l_""$convcell""c_""$batch""b_""$framedims""f_TEST-%j.out" 
                exp_name="$model""_e""$epoch""_""$nlayer""l_""$convcell""c_""$batch""b_""$framedims""f" 
                
                ode_log_file="storage/logs/dataset""$dataset""_""$convcell""c_""$nlayer""l_extrapTrue_f""$framedims""_nruFalse_nru2False_e""$epoch""_b""$batch""_unequalTrue_""$is""_""$os""_""$sample_size"
                nru_log_file="storage/logs/dataset""$dataset""_""$convcell""c_""$nlayer""l_extrapTrue_f""$framedims""_nruTrue_nru2False_e""$epoch""_b""$batch""_unequalTrue_""$is""_""$os""_""$sample_size"
                nru2_log_file="dataset""$dataset""_""$convcell""c_""$nlayer""l_extrapTrue_f""$framedims""_nruFalse_nru2True_e""$epoch""_b""$batch""_unequalTrue_""$is""_""$os""_""$sample_size"
                
                log_file=""
                if [ $model == "VidODE" ]
                then
                    log_file=$ode_log_file
                elif [ $model == "VidODENRU" ]
                then
                    nru="True"
                    log_file=$nru_log_file
                elif [ $model == "VidODENRU2" ]
                then   
                    nru2="True" 
                    log_file=$nru2_log_file
                fi
                
                # Submit training job and get JOB_ID as `train_id` 
                RES=$(sbatch --output $train_output_file --time $train_time --job-name $exp_name ./scripts/train_vidode_job.sh $batch $epoch $framedims $is $os $ckpt $img $nru $nru2 $nlayer $convcell $dataset) 
                train_id=${RES##* }

                # Submit testing job for train_id job only if training was succesfully completed
                RES=$(sbatch --output $test_output_file --time $test_time --dependency=afterok:$train_id ./scripts/test_vidode_job.sh $log_file $dataset) 
                test_id=${RES##* }
                
                # Output job and experiment details to log file
                echo "JOB$jobnum: $train_id""    ""$test_id""    ""$train_output_file""    """$test_output_file"""    ""$log_file"
                echo "$train_id""    ""$test_id""    ""$train_output_file""    """$test_output_file"""    ""$log_file" >> out/job_logs.txt
                jobnum=$((jobnum+1))
                sleep 2
            done
        done
    done
done