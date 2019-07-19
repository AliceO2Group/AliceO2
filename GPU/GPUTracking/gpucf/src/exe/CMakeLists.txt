executables(
    run_gpucf
    cmpcluster
    benchmark
    old_to_new
    load_label
)


add_executable(root_test "root_test.cpp")
target_link_libraries(root_test ${ROOT_LIBRARIES})

add_executable(plot_cluster_per_track "plot_cluster_per_track.cpp")
target_link_libraries(plot_cluster_per_track gpucf ${ROOT_LIBRARIES})

add_executable(testbench 
    "testbench.cpp"
    ${TEST_SRC})
target_link_libraries(testbench gpucf)
