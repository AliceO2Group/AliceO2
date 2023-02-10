#!/bin/bash

if [ "0$O2DPG_ROOT" == "0" ]; then
    echo O2DPG environment not loaded
    exit 1
fi

source $O2DPG_ROOT/DATA/common/gen_topo_helper_functions.sh
