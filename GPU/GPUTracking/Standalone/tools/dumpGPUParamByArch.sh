#!/bin/bash

if [[ -z $3 ]]; then
    echo "Usage: dumpGPUParamByArch.sh [JSON or CSV parameter file] [Architecture] [Output File]"
    exit 1
fi

if ! command -v root &> /dev/null; then
    echo "Cannot run root, please make sure ROOT is available and in the parh"
    exit 1
fi

if [[ ! -f $1 ]]; then
    echo "Input file $1 does not exist"
    exit 1
fi

if [[ -f "include/GPU/GPUDefParametersLoad.inc" ]]; then
    LOADDIR=$(realpath "include/GPU")
elif [[ -f "$O2_ROOT/include/GPU/GPUDefParametersLoad.inc" ]]; then
    LOADDIR=$(realpath "$O2_ROOT/include/GPU/")
else
    echo "Cannot find GPUDefParametersLoad.inc, please run from standalone benchmark folder or set \$O2_ROOT to the standalone or O2 installation"
    exit 1
fi

set -e

TMPDIR=$(mktemp -d)
if [[ $? != 0 ]]; then
    echo "Failed to create a temporary directory"
    exit 1
fi

BASE_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))

if [[ $1 =~ \.csv$ ]]; then
    "${BASE_DIR}"/../../Definitions/Parameters/csv_to_json.sh $1 > "$TMPDIR"/temp.json
    JSON_FILE="$TMPDIR"/temp.json
else
    JSON_FILE=$(realpath $1)
fi

cat <<EOT > "${TMPDIR}"/CMakeLists.txt
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(DumpGPUParam NONE)
include($BASE_DIR/../../Definitions/Parameters/gpu_param_header_generator.cmake)
generate_gpu_param_header("${JSON_FILE}" "$2" "${TMPDIR}/GPUDefParametersDefaultsOnTheFly.h" "${TMPDIR}/GPUDefParametersDefaultsDeviceOnTheFly.h")
EOT

cmake -B "${TMPDIR}" -S"${TMPDIR}"

echo -e "#define GPUCA_GPUTYPE_$2\n" \
        "#define PARAMETER_FILE \"${TMPDIR}/GPUDefParametersDefaultsOnTheFly.h\"\n" \
        "gInterpreter->AddIncludePath(\"${TMPDIR}\");gInterpreter->AddIncludePath(\"${LOADDIR}\");\n" \
        ".x $BASE_DIR/dumpGPUDefParam.C(\"$3\")\n.q\n" | root -l -b

echo -e "\nCreated $3 with parameters for $2 architecture from $1"

rm -Rf "${TMPDIR}"
