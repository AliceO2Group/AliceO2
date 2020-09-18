#!/usr/bin/env bash

#set -e ;# exit on error
set -u ;# exit when using undeclared variable
#set -x ;# debugging

trap kill_benchmark INT

function kill_benchmark() {
  echo "Killing any running benchmark workflows..."
  pkill -9 -f o2-testworkflows-datasampling-benchmark
  exit 1
}

# todo:
#  get stddev out of the monitoring metrics

RESULTS_FILE='data-sampling-benchmark-'$(date +"%y-%m-%d_%H%M")

# Run the benchmark with given parameters
# \param 1 : fractions array name
# \param 2 : payload sizes array name
# \param 3 : number of producers array name
# \param 4 : number of dispatchers array name
# \param 5 : repetitions
# \param 6 : test duration
# \param 7 : warm up cycles - how many first metrics should be ignored (they are sent each 10s)
# \param 8 : test name
# \param 9 : available memory for the benchmark (in MB)
# \param 10: fill - (yes/no) should write zeroes to the produced messages (prevents memory overcommitting)
function benchmark() {

  local fractions_array_name=$1[@]
  local payload_sizes_array_name=$2[@]
  local number_of_producers_array_name=$3[@]
  local number_of_dispatchers_array_name=$4[@]

  local fractions=("${!fractions_array_name}")
  local payload_sizes=("${!payload_sizes_array_name}")
  local number_of_producers=("${!number_of_producers_array_name}")
  local number_of_dispatchers=("${!number_of_dispatchers_array_name}")

  local repetitions=$5
  local test_duration=$6
  local warm_up_cycles=$7
  local test_name=$8
  local available_memory_bytes=$9'000000'
  local fill=${10}

  if [ ! -z $TESTS ] && [[ ! " ${TESTS[@]} " =~ " ${test_name} " ]]; then
    echo "Test '"$test_name"' ignored"
    return
  fi

  echo "======================================================================="
  echo "Running the test '"$test_name"'"

  local test_duration_timeout=$((test_duration + 60))
  local repo_latest_commit=$(git rev-list --format=oneline --max-count=1 HEAD)
  local repo_branch=$(git rev-parse --abbrev-ref HEAD)
  local results_filename='data-sampling-benchmark-'$(date +"%y-%m-%d_%H%M")'-'$test_name
  local test_date=$(date +"%y-%m-%d %H:%M:%S")
  local memory_soft_limit_mbytes=$((available_memory_bytes / 5000000)) # we stop creating new messages at 4/5 of the maximum allowed usage

  printf "DATA SAMPLING BENCHMARK RESULTS\n" > $results_filename
  printf "Test date:              %s\n" "$test_date" >> $results_filename
  printf "Latest commit:          %s\n" "$repo_latest_commit" >> $results_filename
  printf "Branch:                 %s\n" "$repo_branch" >> $results_filename
  printf "Repetitions:            %s\n" "$repetitions" >> $results_filename
  printf "Test duration [s]:      %s\n" "$test_duration" >> $results_filename
  printf "Warm up cycles:         %s\n" "$warm_up_cycles" >> $results_filename
  printf "Available memory [B]:   %s\n" "$available_memory_bytes" >> $results_filename
  printf "Memory soft limit [MB]: %s\n" "$memory_soft_limit_mbytes" >> $results_filename
  echo "fraction       , payload size   , nb producers   , nb dispatchers , messages per second" >> $results_filename

  local common_args="--run -b --infologger-severity info --shm-segment-size "$available_memory_bytes" --test-duration "$test_duration" --throttling "$memory_soft_limit_mbytes
  if [[ $fill == "yes" ]]; then
    common_args=$common_args' --fill'
  fi

  for fraction in ${fractions[@]}; do
    for payload_size in ${payload_sizes[@]}; do
      for nb_producers in ${number_of_producers[@]}; do
        for nb_dispatchers in ${number_of_dispatchers[@]}; do
          for ((rep=0;rep<repetitions;rep++)); do
            echo "************************************************************"
            echo "Launching test for payload size $payload_size bytes, $nb_producers producers, $nb_dispatchers dispatchers, sampling fraction $fraction"

            printf "%15s," "$fraction" >> $results_filename
            printf "%16s," "$payload_size" >> $results_filename
            printf "%16s," "$nb_producers" >> $results_filename
            printf "%16s," "$nb_dispatchers" >> $results_filename

            messages_per_second=
            while [ "$messages_per_second" == 'error' ] || [ -z "$messages_per_second" ]; do
              if [ "$messages_per_second" == 'error' ]; then
                echo "Retrying the test because of an error"
              fi

              # running the benchmark, extracting an array of metrics, ignoring the first warm_up_cycles-1
              # fixme: we assume that the metrics are produced in even (10s) time intervals and all are printed,
              #        we should at least be able notice when something doesn't seem right

              metrics=
              mapfile -t metrics < \
                <( timeout -k 60s $test_duration_timeout o2-testworkflows-datasampling-benchmark $common_args --payload-size $payload_size --producers $nb_producers --dispatchers $nb_dispatchers --sampling-fraction $fraction \
                 | grep -o 'Dispatcher_messages_evaluated,[0-9] [0-9]\{1,\}' \
                 | sed -e 's/Dispatcher_messages_evaluated,[0-9]\{1,\} //'   \
                 | tail -n +$((warm_up_cycles * nb_dispatchers + 1)) )

              pkill -9 -f o2-testworkflows-datasampling-benchmark

              if [ ${#metrics[@]} -ge $(( 2 * nb_dispatchers )) ]; then

                total_start=0
                for ((i = 0; i < nb_dispatchers; i++))
                do
                  (( total_start+=metrics[i] ))
                done

                total_end=0
                for ((i = 1; i < $((1 + nb_dispatchers)); i++))
                do
                  (( total_end+=metrics[-i] ))
                done

                (( messages_per_second = (total_end - total_start) / (${#metrics[@]} / nb_dispatchers - 1 ) ))
                # divide by 10, keeping the last digit
                if [ $messages_per_second -gt 9 ]; then
                  messages_per_second=${messages_per_second:0:-1}.${messages_per_second: -1}
                else
                  messages_per_second=0.$messages_per_second
                fi
              else
                messages_per_second='error'
              fi
            done

            printf "%20s" "$messages_per_second" >> $results_filename
            printf "\n" >> $results_filename

            echo "Dispatcher_messages_evaluated metrics:"
            if [ ${#metrics[@]} -gt 0 ]; then
              printf '%s\n' "${metrics[@]}"
            else
              echo $metrics
            fi
            printf 'Messages per second: %s\n' "${messages_per_second}"
          done
        done
      done
    done
  done
}

function print_usage() {
  echo "Usage: ./o2-datasampling-benchmark.sh [-f] [-m MEMORY_USAGE] [-t TEST]

Run Data Sampling Benchmark and create report files in the directory.

Options:
 -h               Print this message
 -f               Fill messages with zeroes in data procuders. This prevents Linux from overcommitting memory, but
                  slows down the message production rates. It should be used for tests with ZeroMQ. For shmem tests the
                  default protection mechanism is sufficient.
 -m MEMORY_USAGE  Amount of memory (in MB) to be reserved by benchmark. When tested with shared memory, it must be less
                  than the size of /dev/shm. If not specified, it will reserve a half of the free memory at the
                  beginning of the benchmark. The data producers will aim to use less than 4/5 of it. For versions
                  without shared memory enabled, the data producers will aim to leave free one fifth of the amount
                  specified.
 -t TEST          Test name to be run. Use it multiple times to run multiple tests. If not specified, all are run. See
                  the last part of this script to find the available tests.
"
}

MEMORY_USAGE=$(cat /proc/meminfo | grep 'MemFree:' | grep -o "[0-9]\{1,\}")
MEMORY_USAGE=$((MEMORY_USAGE / 1000 / 2))
FILL=no
TESTS=

while getopts 'hfm:t:' option; do
  case "${option}" in
  \?)
    print_usage
    exit 1
    ;;
  h)
    print_usage
    exit 0
    ;;
  f)
    FILL=yes
    ;;
  m)
    MEMORY_USAGE=$OPTARG
    ;;
  t)
    TESTS=("$OPTARG")
    printf '%s\n' "${TESTS[@]}"
    ;;
  esac
done

FRACTIONS=(1.00);
PAYLOAD_SIZE=(16777216 67108864 268435456 1073741824);
NB_PRODUCERS=(8);
NB_DISPATCHERS=(1);
REPETITIONS=1;
TEST_DURATION=300;
WARM_UP_CYCLES=2;
TEST_NAME='memory'

benchmark FRACTIONS PAYLOAD_SIZE NB_PRODUCERS NB_DISPATCHERS $REPETITIONS $TEST_DURATION $WARM_UP_CYCLES $TEST_NAME $MEMORY_USAGE $FILL

FRACTIONS=(0.00 1.00);
PAYLOAD_SIZE=(1 256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456 1073741824);
NB_PRODUCERS=(8);
NB_DISPATCHERS=(1);
REPETITIONS=1;
TEST_DURATION=300;
WARM_UP_CYCLES=6;
TEST_NAME='payloads'

benchmark FRACTIONS PAYLOAD_SIZE NB_PRODUCERS NB_DISPATCHERS $REPETITIONS $TEST_DURATION $WARM_UP_CYCLES $TEST_NAME $MEMORY_USAGE $FILL

FRACTIONS=(0.00 1.00);
PAYLOAD_SIZE=(2097152);
NB_PRODUCERS=(1 2 4 8 16 32);
NB_DISPATCHERS=(1);
REPETITIONS=1;
TEST_DURATION=300;
WARM_UP_CYCLES=6;
TEST_NAME='producers-2MiB'

benchmark FRACTIONS PAYLOAD_SIZE NB_PRODUCERS NB_DISPATCHERS $REPETITIONS $TEST_DURATION $WARM_UP_CYCLES $TEST_NAME $MEMORY_USAGE $FILL

FRACTIONS=(0.0000 0.0001 0.0010 0.0100 0.1000 0.5000 1.0000);
PAYLOAD_SIZE=(2097152);
NB_PRODUCERS=(8);
NB_DISPATCHERS=(1);
REPETITIONS=1;
TEST_DURATION=300;
WARM_UP_CYCLES=6;
TEST_NAME='fractions'

benchmark FRACTIONS PAYLOAD_SIZE NB_PRODUCERS NB_DISPATCHERS $REPETITIONS $TEST_DURATION $WARM_UP_CYCLES $TEST_NAME $MEMORY_USAGE $FILL

FRACTIONS=(0.0000 1.0000);
PAYLOAD_SIZE=(256 2097152);
NB_PRODUCERS=(8);
NB_DISPATCHERS=(1 2 4 8);
REPETITIONS=1;
TEST_DURATION=300;
WARM_UP_CYCLES=6;
TEST_NAME='dispatchers'

benchmark FRACTIONS PAYLOAD_SIZE NB_PRODUCERS NB_DISPATCHERS $REPETITIONS $TEST_DURATION $WARM_UP_CYCLES $TEST_NAME $MEMORY_USAGE $FILL