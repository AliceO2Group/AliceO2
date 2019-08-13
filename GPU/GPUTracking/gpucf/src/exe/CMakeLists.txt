executables(
    benchmark
    cmpcluster
    load_label
    noise_suppression_eval_pipeline
    noise_suppression_minima_epsilon
    old_to_new
    plot_cluster_per_track
    plot_qmaxcutoff
    run_gpucf
    root_test
    testbench
)

add_executable(unittests 
    "unittests.cpp"
    ${TEST_SRC})
target_link_libraries(unittests gpucf)
