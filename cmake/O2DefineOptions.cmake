include_guard()

function(o2_define_options)

  option(BUILD_SHARED_LIBS "Build shared libs" ON)

  option(BUILD_SIMULATION "Build simulation related parts" ON)

  option(BUILD_EXAMPLES "Build examples" ON)

  option(BUILD_TEST_ROOT_MACROS
         "Build the tests toload and compile the Root macros" OFF)

endfunction()
