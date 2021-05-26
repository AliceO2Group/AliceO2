#!/bin/bash -l
set -x

# a script checking logfiles for signs of errors
# can be used to detect most common errors automatically
# returns/exits with 0 if no problem

error=0

for file in "$@"; do
  echo "looking into ${file}"

  fileerrors=0

  # We don't want to see "Error" or "ERROR" messages
  value=$(grep "Error" ${file})
  if [ -n "${value}" ]; then
    echo "check for Error failed"
    let filererrors=fileerrors+1
  fi

  value=$(grep "ERROR" ${file})
  if [ -n "${value}" ]; then
    echo "check for ERROR failed"
    let fileerrors=fileerrors+1
  fi

  if [ "${fileerrors}" != "0" ]; then
    echo "Found problem in file ${file}"
    echo "<--- START OF FILE ${file} ---"
    cat ${file}
    echo "**** END OF FILE ${file} ****>"
  fi

  let error=fileerrors+error
done

exit ${error}
