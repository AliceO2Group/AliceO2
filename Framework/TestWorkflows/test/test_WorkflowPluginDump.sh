#!/bin/sh -e
# Checks that the workflow plugin of an executable, run by o2-dpl-run,
# defines the same workflow as the executable itself.
# usage: test_WorkflowPluginDump.sh <executable> <o2-dpl-run> <library>:<plugin>
strip() { grep -v '"executable"\|"plugin"\|"command"\|"--workflow-plugin"\|"'"$1"'"'; }
exe=$("$1" -b --dump-workflow < /dev/null | strip "$3")
dso=$("$2" -b --workflow-plugin "$3" --dump-workflow < /dev/null | strip "$3")
test -n "$exe"
test "$exe" = "$dso"
test "$("$2" -b --workflow-plugin "$3" --dump-workflow < /dev/null | grep -c "\"plugin\": \"$3\"")" -gt 0
