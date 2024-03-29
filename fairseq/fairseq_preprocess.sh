#!/bin/bash

TRAIN=train;
VALID=dev;
TEST=test;

usage_exit() {
        echo "Usage: $0 [-d DATA] [-v VALID] item ..." 1>&2
        exit 1
}

while getopts d:t:v:e:b:h OPT
do
    case $OPT in
        d)  DATA=$OPTARG
            ;;
        t)  TRAIN=$OPTARG
            ;;
        v)  VALID=$OPTARG
            ;;
        e)  TEST=$OPTARG
            ;;
        b)  BASE=$OPTARG
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done

shift $((OPTIND - 1))

source $BASE/venvs/fairseq/bin/activate;

fairseq-preprocess --source-lang src --target-lang tgt \
--trainpref $DATA/$TRAIN --validpref $DATA/$VALID --testpref $DATA/$TEST \
--destdir ${DATA}_bin --workers 32