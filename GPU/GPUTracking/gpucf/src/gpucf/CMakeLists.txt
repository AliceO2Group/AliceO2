set(GPUCF_SRC
    "./log.cpp"
    "./errorcodes.cpp"

    "./common/Digit.cpp"

    "./io/DigitParser.cpp"
)

add_library(gpucf STATIC ${GPUCF_SRC})
