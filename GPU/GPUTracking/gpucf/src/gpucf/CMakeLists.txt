set(GPUCF_SRC
    "./ClEnv.cpp"
    "./ClusterChecker.cpp"
    "./CsvFile.cpp"
    "./VectorAdd.cpp"
    
    "./common/Cluster.cpp"
    "./common/DataSet.cpp"
    "./common/Digit.cpp"
    "./common/Event.cpp"
    "./common/log.cpp"
    "./common/Object.cpp"
    "./common/RowInfo.cpp"

    "./debug/SCFuzzer.cpp"

    "./executable/Benchmark.cpp"
    "./executable/CfRunner.cpp"
    "./executable/ClCanary.cpp"
    "./executable/CompareClusters.cpp"
    "./executable/Executable.cpp"
    "./executable/SubGroupInfo.cpp"
    "./executable/RunSCFuzzer.cpp"

    "./gpu/GPUAlgorithm.cpp"
    "./gpu/GPUClusterFinder.cpp"
    "./gpu/StreamCompaction.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf 
    ${OpenCL_LIBRARY}
    pthread
)
