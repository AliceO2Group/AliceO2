#!/bin/bash

FULL_CONFIG_JSON='full_config.json'
LDIR=$(pwd)
INPUT_DATA_FILENAME='input_data.txt'
XML_FILENAME='wn.xml'

echo "*** BEGIN Printing environment ***"
env
ulimit -a
echo "*** END Printing environment ***"

# Workaround to remove AliEn libraries. See https://alice.its.cern.ch/jira/browse/JAL-163
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr : \\n | grep -vi 'alien.*2-19' | paste -sd:)

# Workaround for el6 libraries in 8 core queues
# export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr : \\n | grep -vi 'el6-x86_64' | paste -sd:)

### Converting wn.xml to input_data.txt
  sed -rn 's/.*turl="([^"]*)".*/\1/p' $XML_FILENAME > $INPUT_DATA_FILENAME
  if [ ! -s $INPUT_DATA_FILENAME ]; then
    err_message="$INPUT_DATA_FILENAME file is missing, skipping the test execution"
    >&2 echo $err_message; echo $err_message >> .alienValidation.trace
    exit 1
  fi
  test_number_of_input_files=`sed -rn 's/.*turl="alien:\/\/([^"]*)".*/\1/p' $XML_FILENAME | wc -l`
### ^^^ ###

### Creating necessary json files and constructing train test command
  # Reading JSON file
    #inputdata=(`cat $FULL_CONFIG_JSON | jq -r '.inputdata[]'`) # When parsing with for loop
    inputdata=$(cat $FULL_CONFIG_JSON | jq -r '.inputdata | map(select(. != "")) | join(",")')
    
    workflows=$(cat $FULL_CONFIG_JSON | jq -r '[.workflows[]]')
    configurations=`echo $workflows | jq -r '[.[].configuration]'`
    workflow_names=`echo $workflows | jq -r '[.[].workflow_name]'`
    suffixes=`echo $workflows | jq -r '[.[].suffix]'`
    total_tests=`echo $configurations | jq '. | length'`;

    derived_data=`jq -r '.derived_data' $FULL_CONFIG_JSON`
    if [ "$derived_data" == "true" ]; then
      output_configuration=`jq -r '.OutputDirector' $FULL_CONFIG_JSON`
      if [ "$output_configuration" == "null" ]; then
        err_message="OutputDirector tag missing."
        >&2 echo $err_message; echo $err_message >> .alienValidation.trace
        exit 1
      fi
      echo "{ \"OutputDirector\": $output_configuration }" > OutputDirector.json
    fi

  # If input data is empty saving status json as the test is failed
    if [ -z "$inputdata" ]; then
      err_message="The inputdata field is empty."
      >&2 echo $err_message; echo $err_message >> .alienValidation.trace
      exit 1
    fi
### ^^^ ###

cpu_cores=`jq -r '.cpu_cores' $FULL_CONFIG_JSON`
let SHARED_MEMORY=$cpu_cores*1000000000+1500000000
let RATE_LIMIT=$cpu_cores*500000000+500000000
READERS=2
if [ "$cpu_cores" -ge "4" ]; then
  READERS=4
fi

echo "Running on $cpu_cores cores and configuring with shared memory $SHARED_MEMORY, rate limit $RATE_LIMIT, readers $READERS"

### Generating the command
  command=""
  #if [ "$cpu_cores" -eq "8" ]; then
  #  command="echo | "
  #fi
  for (( i=0; i<$total_tests ; i++ )); do
    [[ $total_tests > 1 ]] && index=$((i+1)) || index=""
    # Filenames
      CONFIG_JSON=configuration$index.json
      WORKFLOW_JSON=workflow$index.json
    # Reading the necessary values of json files
      configuration=`echo "$configurations" | jq -r ".[$i]"`
      workflow=`echo $workflows | jq -r "[.[].json.workflow[$i]]"`
      metadata=`echo $workflows | jq -r "[.[].json.metadata[$i]]"`
      workflow_name=`echo "$workflow_names" | jq -r ".[$i]"`
      suffix=`echo "$suffixes" | jq -r ".[$i]"`

    # Writing config and workflow json files
      jq -n "${configuration}" > $CONFIG_JSON
      echo "{\"workflow\": $workflow, \"metadata\": $metadata }" > $WORKFLOW_JSON

    aod_file=""
    #if [ "$i" -eq "0" ]; then
      aod_file="--aod-file @$LDIR/$INPUT_DATA_FILENAME "
      if [ "$derived_data" == "true" ]; then
        aod_file="$aod_file --aod-writer-json OutputDirector.json"
      fi
    #fi

    echo "Adding $workflow_name with configuration $CONFIG_JSON and suffix $suffix"
    command+="$workflow_name -b --configuration json:/$LDIR"/$CONFIG_JSON" --workflow-suffix $suffix --readers $READERS --resources-monitoring 10 --shm-segment-size $SHARED_MEMORY --aod-memory-rate-limit $RATE_LIMIT $aod_file --driver-client-backend stdout:// | "
  done

  # Removing last pipe
  command=${command::-2}
### ^^^ ###

config_files=`ls $LDIR | grep "configuration.*\.json"`
workflow_files=`ls $LDIR | grep "workflow[0-9]*\.json"`
if [ ! -f "$FULL_CONFIG_JSON" ] || [ -z "$workflow_files" ] || [ -z "$config_files" ]; then
  err_message="The configuration file(s) is(are) missing"
  >&2 echo $err_message; echo $err_message >> .alienValidation.trace
  exit 1
fi

# Workaround bug in JAliEn connect
unset ALIEN_PROC_ID

# So far everything is smooth, we can run the test command
START_TIME=`date +%s`
echo "Trying to execute command: ${command}"
echo ""
eval $command
O2_EXITCODE=$?
END_TIME=`date +%s`
let test_wall_time=$END_TIME-$START_TIME

echo ""
echo "O2 exited with $O2_EXITCODE"
echo ""

echo "$O2_EXITCODE" > o2exitcode

if [ "$O2_EXITCODE" -eq "0" ] && [ -s performanceMetrics.json ]; then
  echo ""
  echo "Parsing metrics..."
  # Copied verbatim from grid/prepare_metrics.py
  cat << EOF | /usr/bin/env python - $FULL_CONFIG_JSON $test_wall_time $test_number_of_input_files `hostname -f`

import sys
import copy
import json
import os

with open(sys.argv[1]) as f:
  train = json.load(f)

with open('performanceMetrics.json') as f:
  data = json.load(f)

aggregated = json.loads("{}")

processed = []

def addWorkflow(configs, name):
  global processed
  if not name in aggregated:
    aggregated[name] = {}
  for key in configs:
    # print("K", key)
    processed.append(key)

    for metric in data[key]:
      #print (metric)

      arr = data[key][metric]

      if not metric in aggregated[name]:
        aggregated[name][metric] = copy.deepcopy(arr)
      else:
        if metric == "aod-file-read-info": # strings
          for elem in arr:
            aggregated[name][metric].append(elem)
        else:
          for idx, elem in enumerate(aggregated[name][metric]):
            if idx < len(arr):
              elem["value"] = float(elem["value"]) + float(arr[idx]["value"])

      if metric == "cpuUsedAbsolute":
        cpu_sum = 0
        for elem in arr:
          cpu_sum += float(elem["value"])
        print("CPU seconds of %s: %d"%(key,cpu_sum/1e6))


for workflow in train["workflows"]:
  name = workflow["workflow_name"]
  # print("W", name)
  if not name in aggregated:
    aggregated[name] = { "wagon_id": workflow["wagon_id"] }
  addWorkflow(workflow["configuration"], name)

addWorkflow([ "internal-dpl-aod-global-analysis-file-sink" ], "writer")

reader_list = []
for key in data:
  if not key in processed:
    # print(key)
    reader_list.append(key)
addWorkflow(reader_list, "reader")

# sum everything
full_train = {}
total_read = 0
for key in data:
  for metric in data[key]:
    arr = data[key][metric]

    if metric == "aod-bytes-read-compressed":
      total_read += int(arr[-1]["value"])

    if metric != "aod-file-read-info":
      if not metric in full_train:
        full_train[metric] = copy.deepcopy(arr)
      else:
        for idx, elem in enumerate(full_train[metric]):
          if idx < len(arr):
            elem["value"] = float(elem["value"]) + float(arr[idx]["value"])
aggregated["__full_train__"] = full_train

output = json.loads("{}")
for key in aggregated:
  print(key)

  cpu_sum = 0
  for elem in aggregated[key]["cpuUsedAbsolute"]:
    cpu_sum += int(elem["value"])
  print("CPU seconds: %d"%(cpu_sum/1e6))
  output[key] = {}
  if "wagon_id" in aggregated[key]:
    output[key]["wagon_id"] = aggregated[key]["wagon_id"]
  output[key]["cpu"] = [ cpu_sum/1e6 ]

  for metric in [ "proportionalSetSize" ]:
    print(metric)

    arr = aggregated[key][metric]

    avg = 0
    maxval = 0
    for xy in arr:
      avg += int(xy["value"])
      if maxval < int(xy["value"]):
        maxval = int(xy["value"])

    if len(arr) > 0:
      avg /= len(arr)

    # convert to bytes
    avg *= 1000
    maxval *= 1000

    print("avg = %d B"%avg)
    print("max = %d B"%maxval)

    output[key][metric + "_summary"] = { "avg" : [ avg ] , "max" : [ maxval ] }

output["__full_train__"]["wall"] = [ int(sys.argv[2]) ]
output["__full_train__"]["number_of_input_files"] = [ int(sys.argv[3]) ]
output["__full_train__"]["aod-bytes-read-compressed"] = [ total_read ]
print("Read compressed bytes: %d"%total_read)

output["reader"]["aod-file-read-info"] = aggregated["reader"]["aod-file-read-info"]
# append CE
alien_site = ""
if 'ALIEN_SITE' in os.environ:
  os.environ['ALIEN_SITE']
for elem in output["reader"]["aod-file-read-info"]:
  elem["value"] += ",ce=" + alien_site

output["__full_train__"]["alien_site"] = [ alien_site ]
output["__full_train__"]["host_name"] = [ sys.argv[4] ]

with open('metrics_summary.json', 'w') as outfile:
  json.dump(output, outfile)
  
EOF
  echo ""
fi

#exit $?
exit 0