if(ALICEO2_MODULAR_BUILD)
  #
  # FIXME: this part should disappear when we merge all this new cmake stuff and
  # we change the o2.sh recipe accordingly
  #
  # we use the presence of ALICEO2_MODULAR_BUILD as a signal that we are using
  # the old recipe and we assume BOOST_ROOT is defined and can be used to
  # retrieve the ALIBUILD_BASEDIR
  if(NOT Common_O2_ROOT)
    message(FATAL_ERROR "Don't know how to adapt (yet) to this situation")
  endif()
  get_filename_component(ALIBUILD_BASEDIR ${Common_O2_ROOT}/../.. ABSOLUTE)
  message(STATUS "!!!!")
  message(
    STATUS
      "!!!! Used Common_O2_ROOT location to compute ALIBUILD_BASEDIR=${ALIBUILD_BASEDIR}"
    )

  message(STATUS "!!!! Unsetting most of the -D options and detecting them instead !!!!")
  message(STATUS "!!!! This should be only temporary !!!!")
  message(STATUS "!!!!")

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
