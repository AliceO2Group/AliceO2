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
  MFTSimulation
  SOURCES src/Detector.cxx
          src/DigitizerTask.cxx
  PUBLIC_LINK_LIBRARIES O2::MFTBase
                        ROOT::Physics)

o2_target_root_dictionary(
  MFTSimulation
  HEADERS include/MFTSimulation/Detector.h
          include/MFTSimulation/DigitizerTask.h)

o2_data_file(COPY data DESTINATION Detectors/MFT/simulation)

o2_add_executable(
  digi2raw
  COMPONENT_NAME mft
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
