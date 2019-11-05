set(GPUCF_SRC

    "./algorithms/ClusterFinder.cpp"
    "./algorithms/ClusterFinderConfig.cpp"
    "./algorithms/ClusterFinderProfiler.cpp"
    "./algorithms/ClusterFinderState.cpp"
    "./algorithms/ClusterFinderTest.cpp"
    "./algorithms/cpu.cpp"
    "./algorithms/ReferenceClusterFinder.cpp"
    "./algorithms/StreamCompaction.cpp"
    
    "./common/ClEnv.cpp"
    "./common/Cluster.cpp"
    "./common/ClusterMap.cpp"
    "./common/DataSet.cpp"
    "./common/Digit.cpp"
    "./common/Event.cpp"
    "./common/float.cpp"
    "./common/Kernel1D.cpp"
    "./common/LabelContainer.cpp"
    "./common/log.cpp"
    "./common/MCLabel.cpp"
    "./common/Measurements.cpp"
    "./common/Object.cpp"
    "./common/Position.cpp"
    "./common/RowInfo.cpp"

    "./debug/ClusterChecker.cpp"
    "./debug/DigitDrawer.cpp"
    "./debug/PeakCountTest.cpp"
    "./debug/SCFuzzer.cpp"
    "./debug/utils.cpp"

    "./executable/Benchmark.cpp"
    "./executable/CfRunner.cpp"
    "./executable/CfCLIFlags.cpp"
    "./executable/CompareClusters.cpp"
    "./executable/Executable.cpp"
    "./executable/Testbench.cpp"

    "./experiments/Experiment.cpp"
    "./experiments/TimeCf.cpp"

    "./noisesuppression/NoiseSuppression.cpp"
    "./noisesuppression/NoiseSuppressionOverArea.cpp"
    "./noisesuppression/utils.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf 
    ${OpenCL_LIBRARY}
    pthread
)
