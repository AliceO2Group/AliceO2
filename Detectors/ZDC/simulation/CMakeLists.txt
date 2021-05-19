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
  ZDCSimulation
  SOURCES src/Detector.cxx
          src/Digitizer.cxx
          src/SimCondition.cxx
          src/ZDCSimParam.cxx
          src/SpatialPhotonResponse.cxx
          src/Digits2Raw.cxx
  PUBLIC_LINK_LIBRARIES
    O2::SimulationDataFormat
    O2::ZDCBase
    O2::DataFormatsZDC
    O2::CCDB
    O2::SimConfig
    O2::DetectorsRaw
    O2::Headers)

o2_target_root_dictionary(
  ZDCSimulation
  HEADERS include/ZDCSimulation/Digitizer.h
          include/ZDCSimulation/Detector.h
          include/ZDCSimulation/SimCondition.h
          include/ZDCSimulation/ZDCSimParam.h
          include/ZDCSimulation/SpatialPhotonResponse.h)

o2_data_file(COPY data DESTINATION Detectors/ZDC/simulation)

o2_add_executable(
  digi2raw
  COMPONENT_NAME zdc
  SOURCES src/digi2raw.cxx
  PUBLIC_LINK_LIBRARIES
    O2::ZDCSimulation
    O2::DetectorsRaw
    O2::DetectorsCommonDataFormats
    O2::CommonUtils
    Boost::program_options)
