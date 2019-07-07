# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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

function(o2_show_env var)
  if(DEFINED ENV{${var}})
    file(TO_CMAKE_PATH $ENV{${var}} path)
    message(STATUS "!!!")
    message(STATUS "!!! ${var} is : ")
    foreach(v IN LISTS path)
      message(STATUS "!!! - ${v}")
    endforeach()
  endif()
endfunction()

macro(o2_unset var)
  message(STATUS "!!! Unsetting ${var}=${${var}}")
  unset(${var})
endmacro()

message(STATUS "!!!")
message(STATUS "!!! Using O2RecipeAdapter - this should be only temporary")
message(STATUS "!!!")

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
      "!!! Used Common_O2_ROOT location to compute ALIBUILD_BASEDIR=${ALIBUILD_BASEDIR}"
    )

  message(
    STATUS "!!! Unsetting most of the -D options and detecting them instead")
  message(STATUS "!!!")

  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

  o2_unset(FairRoot_DIR)
  o2_unset(ALICEO2_MODULAR_BUILD)
  o2_unset(ROOTSYS)
  o2_unset(Pythia6_LIBRARY_DIR)
  o2_unset(Geant3_DIR)
  o2_unset(Geant4_DIR)
  o2_unset(VGM_DIR)
  o2_unset(GEANT4_VMC_DIR)
  o2_unset(FAIRROOTPATH)
  o2_unset(BOOST_ROOT)
  o2_unset(DDS_PATH)
  o2_unset(ZMQ_DIR)
  o2_unset(ZMQ_INCLUDE_DIR)
  o2_unset(ALIROOT)
  o2_unset(Protobuf_LIBRARY)
  o2_unset(Protobuf_LITE_LIBRARY)
  o2_unset(Protobuf_PROTOC_LIBRARY)
  o2_unset(Protobuf_INCLUDE_DIR)
  o2_unset(Protobuf_PROTOC_EXECUTABLE)
  o2_unset(GSL_DIR)
  o2_unset(PYTHIA8_INCLUDE_DIR)
  o2_unset(HEPMC3_DIR)
  o2_unset(MS_GSL_INCLUDE_DIR)
  o2_unset(ALITPCCOMMON_DIR)
  o2_unset(Monitoring_ROOT)
  o2_unset(Configuration_ROOT)
  o2_unset(InfoLogger_ROOT)
  o2_unset(Common_O2_ROOT)
  o2_unset(RAPIDJSON_INCLUDEDIR)
  o2_unset(ARROW_HOME)
  o2_unset(benchmark_DIR)
  o2_unset(GLFW_LOCATION)
  o2_unset(CUB_ROOT)

  o2_show_env(LD_LIBRARY_PATH)
  o2_show_env(PATH)

endif()

if(DEFINED ENV{ALIBUILD_O2_TESTS} AND PROJECT_NAME STREQUAL "O2")
  message(STATUS "!!!")
  message(
    STATUS
      "!!! ALIBUILD_O2_TESTS detected. Will patch my tests so they work off the install tree"
    )
  configure_file(${CMAKE_SOURCE_DIR}/tests/tmp-patch-tests-environment.sh.in
                 tmp-patch-tests-environment.sh)
  install(
    CODE [[ execute_process(COMMAND bash tmp-patch-tests-environment.sh) ]])

    install(CODE
            [[ execute_process(COMMAND ldd ${ROOT_rootcling_CMD}) ]])

    install(CODE
            [[ execute_process(COMMAND otool -L ${ROOT_rootcling_CMD}) ]])
endif()

message(STATUS "!!!")
