executables(
    benchmark
    cmpcluster
    load_label
    noise_suppression_eval_pipeline
    old_to_new
    plot_cluster_per_track
    plot_qmaxcutoff
    run_gpucf
    root_test
)

add_executable(testbench 
    "testbench.cpp"
    ${TEST_SRC})
target_link_libraries(testbench gpucf)
