set(GPUCF_SRC
    "./log.cpp"
    "./errorcodes.cpp"
    "./ClEnv.cpp"

    "./common/Digit.cpp"

    "./io/DigitParser.cpp"
    "./io/filename.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
target_link_libraries(gpucf ${OpenCL_LIBRARY})
