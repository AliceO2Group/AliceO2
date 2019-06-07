if(ALIROOT)
  # FIXME: not tested
  add_subdirectory(hough)
endif(ALIROOT)

add_subdirectory(aliceHLTwrapper)
add_subdirectory(O2MessageMonitor)
add_subdirectory(DataFlow)
add_subdirectory(Publishers)
add_subdirectory(DataCompression)
add_subdirectory(Mergers)

# * add_subdirectory (O2Device) // already included in src/CMakeLists.txt as it
#   is needed earlier
# * add_subdirectory (PCG) // already included in src/CMakeLists.txt as it is
#   needed earlier
