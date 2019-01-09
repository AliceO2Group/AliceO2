set(GPUCF_SRC
    "./log.cpp"
    "./ClCanary.cpp"
    "./ClEnv.cpp"
    "./Executable.cpp"
    "./VectorAdd.cpp"

    "./common/Digit.cpp"

    "./io/DigitParser.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf ${OpenCL_LIBRARY})
