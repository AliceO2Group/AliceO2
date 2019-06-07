macro(o2_setup_flags)

  if(ENABLE_CASSERT) # For the CI, we want to have <cassert> assertions enabled
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  endif()

  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

  # make sure Debug build not optimized (does not seem to work without CACHE +
  # FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG
      "-g -O0"
      CACHE STRING "Debug mode build flags" FORCE)
  set(CMAKE_C_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG}"
      CACHE STRING "Debug mode build flags" FORCE)

endmacro()
