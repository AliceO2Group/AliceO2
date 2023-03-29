#!/bin/bash

if [ "0$O2DPG_ROOT" == "0" ]; then
    echo "O2DPG environment not loaded" 1>&2
    exit 1
fi

source $O2DPG_ROOT/DATA/production/workflow-setup.sh || { echo "workflow-setup.sh failed" 1>&2 && exit 1; }
