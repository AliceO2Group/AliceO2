macro(o2_setup_coverage)
  # Build type for coverage builds
  set(CMAKE_CXX_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE}")
  set(CMAKE_Fortran_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
  set(CMAKE_LINK_FLAGS_COVERAGE "--coverage -fprofile-arcs  -fPIC")

  mark_as_advanced(CMAKE_CXX_FLAGS_COVERAGE
                   CMAKE_C_FLAGS_COVERAGE
                   CMAKE_Fortran_FLAGS_COVERAGE
                   CMAKE_LINK_FLAGS_COVERAGE)
endmacro()

