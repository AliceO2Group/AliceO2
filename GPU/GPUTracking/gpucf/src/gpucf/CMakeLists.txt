set(GPUCF_SRC
    "./log.cpp"
    "./CfRunner.cpp"
    "./ClCanary.cpp"
    "./ClEnv.cpp"
    "./Executable.cpp"
    "./VectorAdd.cpp"
    "./GPUClusterFinder.cpp"

    "./common/Digit.cpp"
    "./common/Cluster.cpp"

    "./io/DigitParser.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf ${OpenCL_LIBRARY})
