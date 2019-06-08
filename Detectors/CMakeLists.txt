# * add_subdirectory(Base) # already done earlier in src/CMakeLists.txt

add_subdirectory(Passive) # must be first as some detector's macros use it

add_subdirectory(PHOS)
add_subdirectory(CPV)
add_subdirectory(EMCAL)
add_subdirectory(FIT)
add_subdirectory(HMPID)
add_subdirectory(TOF)
add_subdirectory(ZDC)

# already done earlier as they are required for GPU
# and GPU is required for Detectors/TPC ...
# * add_subdirectory(ITSMFT)
# * add_subdirectory(TRD) 

add_subdirectory(MUON)

add_subdirectory(TPC)

add_subdirectory(GlobalTracking)
add_subdirectory(GlobalTrackingWorkflow)

if(BUILD_SIMULATION)
  add_subdirectory(gconfig)
endif()

o2_file(COPY Geometry gconfig DESTINATION Detectors/)
