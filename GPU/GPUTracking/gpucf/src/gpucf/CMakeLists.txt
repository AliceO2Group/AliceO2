set(GPUCF_SRC
    "./log.cpp"
    "./ChargeMap.cpp"
    "./DataSet.cpp"
    "./ClEnv.cpp"
    "./ClusterChecker.cpp"
    "./GPUClusterFinder.cpp"
    "./Object.cpp"
    "./RowInfo.cpp"
    "./VectorAdd.cpp"
    
    "./common/Digit.cpp"
    "./common/Cluster.cpp"

    "./executable/CfRunner.cpp"
    "./executable/ClCanary.cpp"
    "./executable/Executable.cpp"
    "./executable/SubGroupInfo.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf 
    ${OpenCL_LIBRARY}
    pthread
)
