include_guard()

function(o2_define_options)

  option(BUILD_SHARED_LIBS "Build shared libs" ON)

  option(BUILD_SIMULATION "Build simulation related parts")

  option(BUILD_EXAMPLES "Build examples")

  option(BUILD_TEST_ROOT_MACROS
         "Build the tests toload and compile the Root macros" OFF)

  option(
    BUILD_FOR_DEV
    "Build targeted at developpers - RPATH is fast on build tree but slow on install"
    OFF)

endfunction()
