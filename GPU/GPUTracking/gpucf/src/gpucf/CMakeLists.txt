set(GPUCF_SRC
    "./ClusterChecker.cpp"
    
    "./common/ClEnv.cpp"
    "./common/Cluster.cpp"
    "./common/ClusterMap.cpp"
    "./common/DataSet.cpp"
    "./common/Digit.cpp"
    "./common/DigitDivider.cpp"
    "./common/Event.cpp"
    "./common/float.cpp"
    "./common/Fragment.cpp"
    "./common/Kernel1D.cpp"
    "./common/log.cpp"
    "./common/Measurements.cpp"
    "./common/Object.cpp"
    "./common/Position.cpp"
    "./common/RowInfo.cpp"

    "./debug/SCFuzzer.cpp"

    "./executable/Benchmark.cpp"
    "./executable/CfRunner.cpp"
    # "./executable/ClCanary.cpp"
    "./executable/CompareClusters.cpp"
    "./executable/Executable.cpp"
    # "./executable/SubGroupInfo.cpp"
    # "./executable/RunSCFuzzer.cpp"

    "./experiments/Experiment.cpp"
    "./experiments/TimeCf.cpp"

    "./algorithms/ClusterFinder.cpp"
    "./algorithms/ClusterFinderConfig.cpp"
    "./algorithms/ClusterFinderProfiler.cpp"
    "./algorithms/ClusterFinderState.cpp"
    "./algorithms/ClusterFinderTest.cpp"
    # "./algorithms/VectorAdd.cpp"
    "./algorithms/ReferenceClusterFinder.cpp"
    "./algorithms/StreamCompaction.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf 
    ${OpenCL_LIBRARY}
    pthread
)
