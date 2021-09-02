#!/bin/bash

if [ "0$O2DATAPROCESSING_ROOT" == "0" ]; then
    echo O2DataProcessing environment not loaded
    exit 1
fi

source $O2DATAPROCESSING_ROOT/common/setenv.sh
