#!/bin/bash

WORKDIR=`pwd`

LDIR=`dirname "${BASH_SOURCE[0]}"`
cd $LDIR

./extractDataModel.py > $WORKDIR/htmloutput.txt

DOCUMENTATIONDIR=$WORKDIR/analysis-framework/docs/datamodel
./mdUpdate.py 1 $WORKDIR/htmloutput.txt $DOCUMENTATIONDIR/ao2dTables.md $DOCUMENTATIONDIR/ao2dTables.md
./mdUpdate.py 2 $WORKDIR/htmloutput.txt $DOCUMENTATIONDIR/helperTaskTables.md $DOCUMENTATIONDIR/helperTaskTables.md
./mdUpdate.py 3 $WORKDIR/htmloutput.txt $DOCUMENTATIONDIR/pwgTables.md $DOCUMENTATIONDIR/pwgTables.md
./mdUpdate.py 4 $WORKDIR/htmloutput.txt $DOCUMENTATIONDIR/joinsAndIterators.md $DOCUMENTATIONDIR/joinsAndIterators.md
