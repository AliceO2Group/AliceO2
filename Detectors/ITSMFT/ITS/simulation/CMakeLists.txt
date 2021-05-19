# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(
  ITSSimulation
  SOURCES src/V11Geometry.cxx
          src/V1Layer.cxx
          src/V3Layer.cxx
          src/Detector.cxx
          src/V3Services.cxx
  PUBLIC_LINK_LIBRARIES
    O2::ITSBase
    O2::ITSMFTSimulation
    ROOT::Physics)

o2_target_root_dictionary(
  ITSSimulation
  HEADERS include/ITSSimulation/Detector.h
          include/ITSSimulation/V1Layer.h
          include/ITSSimulation/V3Layer.h
          include/ITSSimulation/V11Geometry.h
          include/ITSSimulation/V3Services.h)

o2_data_file(COPY data DESTINATION Detectors/ITS/simulation)

o2_add_executable(
  digi2raw
  COMPONENT_NAME its
  TARGETVARNAME itsdigi2raw_exe
  SOURCES src/digi2raw.cxx
  PUBLIC_LINK_LIBRARIES
    O2::ITSMFTReconstruction
    O2::DataFormatsITSMFT
    O2::ITSMFTBase
    O2::ITSMFTSimulation
    O2::DetectorsRaw
    O2::DetectorsCommonDataFormats
    O2::CommonUtils
    Boost::program_options)

if(NOT
   APPLE)

  set_property(
    TARGET ${itsdigi2raw_exe}
    PROPERTY LINK_WHAT_YOU_USE
             ON)

endif()
