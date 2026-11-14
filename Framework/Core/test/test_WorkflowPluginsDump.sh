#!/bin/sh -e
# Dumps the workflow of the two test plugins and checks that it records,
# for each data processor, the plugin which defines it.
# usage: test_WorkflowPluginsDump.sh <o2-test-framework-WorkflowPlugins>
A=O2TestFrameworkWorkflowPluginAPlugin:TestFrameworkWorkflowPluginA
B=O2TestFrameworkWorkflowPluginBPlugin:TestFrameworkWorkflowPluginB
dump=$("$1" -b --workflow-plugin $A,$B --dump-workflow < /dev/null)
echo "$dump" | grep -q "\"plugin\": \"$A\""
echo "$dump" | grep -q "\"plugin\": \"$B\""
test "$(echo "$dump" | grep -c '"plugin"')" -eq 2
