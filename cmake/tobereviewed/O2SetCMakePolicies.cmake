# FIXME: most of those are the default for CMake 3.14.
# no longer needed then.

function(o2_set_cmake_policies)
  foreach(p
          CMP0025
          CMP0028
          CMP0042
          CMP0057
          CMP0066
          CMP0067
          CMP0068
          CMP0074
          CMP0077)
    if(POLICY ${p})
      cmake_policy(SET ${p} NEW)
    endif()
  endforeach()
endfunction()

