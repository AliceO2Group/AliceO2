#!/bin/bash -l
set -x

# a script checking logfiles for signs of error
# can be used to detect most common errors automatically
# returns/exits with 0 if no problem

error=0

for file in "$@"; do
  echo "looking into ${file}"

  # We don't want to see "Error" or "ERROR" messages
  value=$(grep "Error" ${file})
  if [ -n "${value}" ]; then
    echo "check for Error failed"
    let error=error+1
  fi

  value=$(grep "ERROR" ${file})
  if [ -n "${value}" ]; then
    echo "check for ERROR failed"
    let error=error+1
  fi

done
  
exit ${error}
