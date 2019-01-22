set(GPUCF_SRC
    "./log.cpp"
    "./ChargeMap.cpp"
    "./ClEnv.cpp"
    "./ClusterChecker.cpp"
    "./GPUClusterFinder.cpp"
    "./RowInfo.cpp"
    "./VectorAdd.cpp"

    "./common/Digit.cpp"
    "./common/Cluster.cpp"

    "./executable/CfRunner.cpp"
    "./executable/ClCanary.cpp"
    "./executable/Executable.cpp"

    "./io/ClusterParser.cpp"
    "./io/ClusterWriter.cpp"
    "./io/DigitParser.cpp"
    "./io/DigitWriter.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf 
    ${OpenCL_LIBRARY}
    pthread
    )
