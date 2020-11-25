#!/bin/bash
USER="iithendaraa.subramanian" 
HOSTS="pascal turing newton feynman faraday"
DOMAIN="livia.etsmtl.ca"
QUERY_1="nvidia-smi --query-gpu=name,utilization.gpu,memory.total,memory.used --format=csv,noheader"
QUERY_2="echo -n \"CPU usage: \" ; top -bn1 | head -n 3 | tail -n 1 | cut -f 2 -d \" \""
QUERY_3="free -m"
for HOSTNAME in ${HOSTS} ; do
    printf "\n############### $HOSTNAME ###############\n"
    for QUERY in "$QUERY_1" "$QUERY_2" "$QUERY_3" ; do
        ssh -l $USER $HOSTNAME.$DOMAIN $QUERY
    done
done