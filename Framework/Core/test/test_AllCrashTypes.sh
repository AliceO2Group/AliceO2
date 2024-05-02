#!/bin/sh -e
echo $PATH
printf "ok\nTesting runtime-init..."
o2-framework-crashing-workflow --crash-type=runtime-init --completion-policy=quit -b --run | grep -q "Exception caught: This is a std::runtime_error" || { printf "runtime error not found" ; exit 1; }
printf "ok\nTesting framework-init..."
o2-framework-crashing-workflow --crash-type=framework-init --completion-policy=quit -b --run | grep -q "Exception caught: This is a o2::framework::runtime_error" || { printf "framework error not found" ;  exit 1; }
printf "ok\nTesting framework-run..."
o2-framework-crashing-workflow --crash-type=framework-run --completion-policy=quit -b --run | grep -q "Unhandled o2::framework::runtime_error reached the top of main of o2-framework-crashing-workflow, device shutting down. Reason: This is a o2::framework::runtime_error" || { printf "framework error not found" ; exit 1; }
printf "ok\nTesting runtime-run..."
o2-framework-crashing-workflow --crash-type=runtime-run --completion-policy=quit --run | grep -q "Unhandled o2::framework::runtime_error reached the top of main of o2-framework-crashing-workflow, device shutting down. Reason: This is a std::runtime_error" || { echo "runtime error not found" ; exit 1; }
printf "ok\n"

export O2_NO_CATCHALL_EXCEPTIONS=1
echo O2_NO_CATCHALL_EXCEPTIONS enabled
printf "ok\nTesting runtime-init..."
o2-framework-crashing-workflow --crash-type=runtime-init --completion-policy=quit -b --run | grep -v -q "Exception caught: This is a std::runtime_error" || { printf "runtime error not found" ; exit 1; }
printf "ok\nTesting framework-init..."
o2-framework-crashing-workflow --crash-type=framework-init --completion-policy=quit -b --run | grep -v -q "Exception caught: This is a o2::framework::runtime_error" || { printf "framework error not found" ;  exit 1; }
printf "ok\nTesting framework-run..."
o2-framework-crashing-workflow --crash-type=framework-run --completion-policy=quit -b --run | grep -v -q "Unhandled o2::framework::runtime_error reached the top of main of o2-framework-crashing-workflow, device shutting down. Reason: This is a o2::framework::runtime_error" || { printf "framework error not found" ; exit 1; }
printf "ok\nTesting runtime-run..."
o2-framework-crashing-workflow --crash-type=runtime-run --completion-policy=quit --run | grep -v -q "Unhandled o2::framework::runtime_error reached the top of main of o2-framework-crashing-workflow, device shutting down. Reason: This is a std::runtime_error" || { echo "runtime error not found" ; exit 1; }
printf "ok"
