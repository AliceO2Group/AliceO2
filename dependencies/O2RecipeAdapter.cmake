# FIXME: this part should disappear when we merge all this new cmake stuff and
# we change the o2.sh recipe accordingly.
#
# We "adapt" two things here :
#
# 1. we unset most of the -D variables that were passed to cmake so our auto-
#   detection has a chance to work. Should not be needed in the long run if we
#   use the correct -D set from the beginning
#
# 1. we patch those tests that require some environment (most notably the O2_ROOT
#   variable) to convert from O2_ROOT pointing to build tree to O2_ROOT pointing
#   to install tree. Should not be needed in the long run if we consider (as we
#   should, I would argue) that tests are running off the build tree, before
#   installation (and are not installed, as there's probably no point in doing
#   so) Should not be needed in the long run if we consider (as we should, I
#   would argue) that tests are running off the build tree, before installation
#   (and are not installed, as there's probably no point in doing so)
#

message(STATUS "!!!!")
message(STATUS "!!!! Using O2ReciperAdapter - this should be only temporary")
message(STATUS "!!!!")

if(ALICEO2_MODULAR_BUILD)
  #
  # we use the presence of ALICEO2_MODULAR_BUILD as a signal that we are using
  # the old recipe and we assume Common_O2_ROOT is defined and can be used to
  # retrieve the ALIBUILD_BASEDIR
  #
  if(NOT Common_O2_ROOT)
    message(FATAL_ERROR "Don't know how to adapt (yet) to this situation")
  endif()
  get_filename_component(ALIBUILD_BASEDIR ${Common_O2_ROOT}/../.. ABSOLUTE)
  message(
    STATUS
      "!!!! Used Common_O2_ROOT location to compute ALIBUILD_BASEDIR=${ALIBUILD_BASEDIR}"
    )

  message(
    STATUS "!!!! Unsetting most of the -D options and detecting them instead")

  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

  unset(FairRoot_DIR)
  unset(ALICEO2_MODULAR_BUILD)
  unset(ROOTSYS)
  unset(Pythia6_LIBRARY_DIR)
  unset(Geant3_DIR)
  unset(Geant4_DIR)
  unset(VGM_DIR)
  unset(GEANT4_VMC_DIR)
  unset(FAIRROOTPATH)
  unset(BOOST_ROOT)
  unset(DDS_PATH)
  unset(ZMQ_DIR)
  unset(ZMQ_INCLUDE_DIR)
  unset(ALIROOT)
  unset(Protobuf_LIBRARY)
  unset(Protobuf_LITE_LIBRARY)
  unset(Protobuf_PROTOC_LIBRARY)
  unset(Protobuf_INCLUDE_DIR)
  unset(Protobuf_PROTOC_EXECUTABLE)
  unset(GSL_DIR)
  unset(PYTHIA8_INCLUDE_DIR)
  unset(HEPMC3_DIR)
  unset(MS_GSL_INCLUDE_DIR)
  unset(ALITPCCOMMON_DIR)
  unset(Monitoring_ROOT)
  unset(Configuration_ROOT)
  unset(InfoLogger_ROOT)
  unset(Common_O2_ROOT)
  unset(RAPIDJSON_INCLUDEDIR)
  unset(ARROW_HOME)
  unset(benchmark_DIR)
  unset(GLFW_LOCATION)
  unset(CUB_ROOT)

endif()

if(DEFINED ENV{ALIBUILD_O2_TESTS})
  message(STATUS "!!!!")
  message(
    STATUS
      "!!!! ALIBUILD_O2_TESTS detected. Will patch my tests so they work off the install tree"
    )
  configure_file(${CMAKE_SOURCE_DIR}/tests/tmp-patch-tests-environment.sh.in
                 tmp-patch-tests-environment.sh)
  install(
    CODE [[ execute_process(COMMAND bash tmp-patch-tests-environment.sh) ]])
endif()
message(STATUS "!!!!")
