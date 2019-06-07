include_guard()

function(o2_check_cxx_features)
  # FIXME: missing the make_unique here compared to previous version
  foreach(FEAT "cxx_aggregate_default_initializers" "cxx_binary_literals"
          "cxx_generic_lambdas" "cxx_user_literals")
    if(NOT "${FEAT}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
      message(FATAL_ERROR "We miss ${FEAT} feature with this compiler")
    endif()
  endforeach()
endfunction()
