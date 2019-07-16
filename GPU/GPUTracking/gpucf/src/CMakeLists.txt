add_subdirectory(gpucf)
add_subdirectory(test)

add_executable(run_gpucf "run_gpucf.cpp")
target_link_libraries(run_gpucf gpucf)

# add_executable(clcanary "clcanary.cpp")
# target_link_libraries(clcanary gpucf)

# add_executable(subgroupinfo "subgroupinfo.cpp")
# target_link_libraries(subgroupinfo gpucf)

add_executable(cmpcluster "cmpcluster.cpp")
target_link_libraries(cmpcluster gpucf)

add_executable(benchmark "benchmark.cpp")
target_link_libraries(benchmark gpucf)

# add_executable(scfuzzer "scfuzzer.cpp")
# target_link_libraries(scfuzzer gpucf)

add_executable(args_test "args_test.cpp")

add_executable(old_to_new "old_to_new.cpp")
target_link_libraries(old_to_new gpucf)

add_executable(load_label "load_label.cpp")
target_link_libraries(load_label gpucf)

add_executable(testbench 
    "testbench.cpp"
    ${TEST_SRC})
target_link_libraries(testbench gpucf)
