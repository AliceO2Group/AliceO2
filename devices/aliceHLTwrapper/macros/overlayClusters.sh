#! /bin/bash

printHelp=0
merge_nEvents=0
targetdir=
nOutput=0
outputBaseName=event

scriptname=`basename $0`
scriptdir=`dirname $0`
macroname=overlayClusters.C
macropath=$scriptdir/$macroname

if ! test -e "$macropath" ; then
cat<<EOF

Error can not find macro '$macropath'; has to be in the same folder as script

EOF
  printHelp=1

else
  [ "$#" -ge 1 ] && merge_nEvents=$1
  shift
  [ "$#" -ge 1 ] && targetdir=$1
  shift
  [ "$#" -ge 1 ] && outputBaseName=$1
  shift
  [ "$merge_nEvents" -eq 0 ] && printHelp=1
fi


if [ "$printHelp" -gt 0 ]; then
cat<<EOF
$scriptname: merge TPC cluster data files with help of root macro $macroname
Usage:
   ls path_to_event_folder | $scriptname count targetdir basename

   count        number of files to merge
   targetdir    target directory [optional]
   basename     basename of folders for merged data samples [optional]

The input event folders are read from standard input.

A working aliroot setup is required in order to start root and load the required
AliRoot libraries.

Current script/macro path:
  $scriptdir
EOF
exit
fi

c=0
unset inputeventlist
while read inputevent; do
  inputeventlist=(${inputeventlist[@]} $inputevent)
  if [ "$c" -ge $(( merge_nEvents - 1 )) ]; then
    outputevent=`echo $targetdir | sed -e 's|\([^/]\)$|\1/|'`$outputBaseName`printf %03d $nOutput`
    echo merging ...
    for f in ${inputeventlist[@]}; do echo "   $f"; done
    echo "  -> $outputevent"

    mkdir -p $outputevent || break
    # for each event to be generated loop over all slices and partitions
    # and merge the files with corresponding specification
    for ((slice=0; slice<36; slice++)); do
      for ((part=0; part<6; part++)); do
	# the data specification of HLT TPC data blocks describes ranges of
	# from-to slices/partitions in the corresponding bitfields of the 32bit specification
	spec=`printf 0x%02x%02x%02x%02x $slice $slice $part $part`
	echo "$outputevent: processing slice $slice partition $part (data specification $spec)"
	for inputpath in `ls ${inputeventlist[0]}/*$spec* | sed -e '/CLUSTERS/!d'`; do
	  filename=`basename $inputpath`
	  outputfile=$outputevent/$filename
	  if test -e $outputfile ; then
	    echo "file $outputfile existing, skipping ..."
	    continue
	  fi
	  echo ${inputeventlist[@]} > $outputevent/filelist.txt
	  (echo "-o $outputevent/$filename"; for ev in ${inputeventlist[@]}; do echo "$ev/$filename"; done) \
	      | root -b -q -l $macropath
	done
      done
    done

    let nOutput++
    c=0
    unset inputeventlist
  else
    let c++
  fi
done
