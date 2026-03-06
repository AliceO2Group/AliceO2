#!/usr/bin/env bash

[[ -z $1 ]] && { echo "Usage: csv_to_json.sh CSV_FILE"; exit 1; }

LANG=C
LC_ALL=C
DELIM=$'\xFF'
set -o pipefail
sed -E \
  ':loop
   s/^(([^"]*"[^"]*")*[^"]*),/\1'$DELIM'/;
   t loop' \
  $1 | \
awk -F$DELIM \
  'BEGIN {
     print "{"
   } {
     if (count == 0) {
       for (i = 1; i <= NF; i++) {
         names[i] = $i
       }
     } else if ($1 == "CORE:" || $1 == "LB:" || $1 == "PAR:") {
       if (paramprinted) print "\n    }"
       else if (lineprinted) print ""
       if (catprinted) print "  },"
       lineprinted = 0
       paramprinted = 0
       catprinted = 1
       gsub(/:$/, "", $1)
       print "  \""$1"\": {";
     } else if ($1 != "") {
       if (lineprinted) print ""
       if (paramprinted) print "    },"
       lineprinted = 0
       paramprinted = 1
       print "    \""$1"\": {";
       lineprinted = 0
       for (i=2; i<=NF; i++) {
         if ($i != "") {
           gsub(/^"/, "", $i)
           gsub(/"$/, "", $i)
           gsub(/""/, "\"", $i)
           if (lineprinted) print ","
           lineprinted = 1
           printf("      \"%s\": %s", names[i], $i)
         }
       }
     }
     count++;
   } END {
     if (paramprinted) print "\n    }"
     if (catprinted) print "  }"
     print "}"
   }'
