#!/bin/sh

get_cmd_write_digits() {
	if [ $# -eq 0 ]; then
		echo "must at least specify the destination of the dump"
		return
	fi
	local dest=$1
	local output=$2
  local format=${3:-missing}
	local opt=""
	local cmd=""

	if [ "$dest" = "binary" ]; then
		opt="--binary-file-format $format"
	elif [ "$dest" = "text" ]; then
		opt="--txt"
	elif [ "$dest" = "screen" ]; then
		opt="--txt"
	else
		echo "dest=$dest is not one of {binary,text,screen}"
		return
	fi

	cmd="o2-mch-digits-writer-workflow -b "
	cmd+=" $opt"
	if [ $output ]; then
		cmd+=" --outfile $output"
	else
		cmd+=" --no-file"
	fi
	echo $cmd
}

run_workflow() {
	local cmd=$1
	local msg=$2
	local verbose=$3

  if [ ! -z $VERBOSE ]; then
    verbose="verbose"
  fi

	cmd+=" | o2-dpl-run --run -b"

	echo "#############"
	echo "# $2"
	echo "$cmd"
	echo "#############"

	if [ "$verbose" = "verbose" ]; then
		eval $cmd
	else
		eval $cmd &>/dev/null
	fi
}

generate_digits() {
	local NROFPERTOF=${1:-1}
	local NTF=${2:-1}
	local OCC=${3:-0.0001}
  local SEED=${4:-1}
  local digitFile=${5:-UNKNOWN}
  local format=${6:-3}
	local cmd=""

	cmd="o2-mch-digits-random-generator-workflow -b "
	cmd+="--nof-rofs-per-tf $NROFPERTF "
	cmd+="--max-nof-tfs $NTF "
	cmd+="--occupancy $OCC "
	cmd+="--seed $SEED "
	cmd+="| "
	cmd+=$(get_cmd_write_digits binary $digitFile $format)
 
	run_workflow $cmd "Generate random digits and save them" verbose
}

dump_digits() {
	if [ $# -eq 0 ]; then
		echo "must specify the name of the digit file to dump"
		return
	fi
	local digitFile=${1:-digits.dump}
	local dest=${2:-invalid}
	local output=$3
  local format=$4
	local cmd=""
  local verbose=""

  if [ "$dest" = "screen" ]; then
    verbose=verbose
  fi

	cmd="o2-mch-digits-file-reader-workflow -b --infile $digitFile | "
	cmd+=$(get_cmd_write_digits $dest $output $format)

	run_workflow $cmd "Dump digits" $verbose
}

create_ctf() {
	if [ $# -eq 0 ]; then
		echo "must specify the name of the digit file to compress"
		return
	fi
	local digitsFile=$1
	local outputType=${2:-ctf}
	local cmd=""

	cmd="o2-mch-digits-file-reader-workflow --infile ${digitsFile} -b | "
	cmd+="o2-mch-entropy-encoder-workflow -b |"
	cmd+="o2-ctf-writer-workflow --onlyDet MCH --no-grp --output-type $outputType -b"

	run_workflow $cmd "Read digits and create $outputType"
}

get_ctfs() {
	if [ $# -eq 0 ]; then
		echo "must specify the number of tfs to look for"
		return
	fi
	local ntf=$1
	local ctfs=""
  local run=0
  local orbit=0

	for tf in $(seq 0 $(($ntf - 1))); do
		ctfs+=$(printf "o2_ctf_run%08d_orbit%010d_tf%010d.root " $run $orbit $tf)
	done
	ctfs=$(echo $ctfs | sed 's/.$//') # remove trailing coma

	echo "$ctfs"
}

read_ctf() {
	if [ $# -lt 2 ]; then
		echo "must specify the CTF(s) to read and output digit filename"
		return
	fi
	local ctfs=$1
	local ctfdigitfile=$2
  local format=$3
	local cmd=""

	cmd="o2-ctf-reader-workflow --onlyDet MCH --ctf-input ${ctfs} -b | "
	cmd+=$(get_cmd_write_digits binary $ctfdigitfile $format)

	run_workflow $cmd "Read CTF and write digits"
}

NROFPERTF=${1:-128}
NTF=${2:-1}
OCC=${3:-0.1}
SEED=${4:-1}
BINARY_FILE_FORMAT=${5:-3}
textDump=0

COUNT=$(( $NROFPERTF * $NTF * $OCC * 1064008 ))
COUNT=$(echo $COUNT | awk '{print int($0)}')

if [ $COUNT -lt 100000 ]; then
  # activate text dump if we don't have too many digits
  textDump=1 
fi

refdigits=digits_ref_rof_${NROFPERTF}_tf_${NTF}_occ_${OCC}_seed_${SEED}

# Generate digits
generate_digits $NROFPERTF $NTF $OCC $SEED $refdigits.orig.data $BINARY_FILE_FORMAT

# Dump the digits to verify sample-sink couple
dump_digits ${refdigits}.orig.data binary ${refdigits}.check.data $BINARY_FILE_FORMAT

echo '# Remove dictionary data if any'
rm -f ctf_dictionary.root

# read Digits and create CTF(s) files
create_ctf ${refdigits}.orig.data

ctfs=$(get_ctfs $NTF)

# read CTF and write digits
read_ctf $ctfs ${refdigits}.ctf.data $BINARY_FILE_FORMAT

# # Dump the digits on screen'
# dump_digits ${refdigits}.ctf.data screen

mkdir nodict && pushd nodict

# Read digits and encode them, but only write out the frequencies dictionary
create_ctf ../${refdigits}.orig.data dict

# Read digits and encode them in a CTF with external dict
create_ctf ../${refdigits}.orig.data

ctfs=$(get_ctfs $NTF)

# read CTF and write digits
read_ctf $ctfs ${refdigits}.nodict.ctf.data $BINARY_FILE_FORMAT

popd

if [ $textDump -eq 1 ]; then
# Make a text version of all digit files
  for d in $(find . -name "digits*"); 
  do
    dump_digits $d text $d.txt
  done
fi

echo '# Compare digits before after ctf encoding/decoding'
if [ $textDump -eq 1 ]; then
  echo '# Text versions'
  find . -name "digits*.txt" -exec shasum -a 256 {} \; | sort
fi
echo '# Binary versions'
find . -name "digits*.data" -exec shasum -a 256 {} \; | sort

echo '# Sizes'
find . -name '*ctf_*' -exec ls -alh {} \;
find . -name 'digits*.data' -exec ls -alh {} \;

