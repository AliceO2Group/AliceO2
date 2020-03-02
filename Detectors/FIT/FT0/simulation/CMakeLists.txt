# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FT0Simulation
  SOURCES src/Detector.cxx
  SOURCES src/Digitizer.cxx
  SOURCES src/Digits2Raw.cxx

  PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::Framework
  O2::FT0Base
  O2::DataFormatsFT0
  O2::DetectorsRaw
  O2::Headers)

o2_target_root_dictionary(FT0Simulation HEADERS
  include/FT0Simulation/Detector.h
  include/FT0Simulation/Digitizer.h
  include/FT0Simulation/Digits2Raw.h
  include/FT0Simulation/DigitizationParameters.h
   )
