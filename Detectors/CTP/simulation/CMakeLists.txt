# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CTPSimulation
  SOURCES src/Digitizer.cxx
  src/Digits2Raw.cxx
  PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::Framework
  O2::DataFormatsCTP
  O2::DetectorsRaw
  O2::Headers)
o2_target_root_dictionary(CTPSimulation HEADERS
  include/CTPSimulation/Digitizer.h)
o2_add_executable(digi2raw
                  COMPONENT_NAME ctp
                  SOURCES src/digi2raw.cxx
                  PUBLIC_LINK_LIBRARIES O2::CTPSimulation
                                        O2::DetectorsRaw
                                        O2::DetectorsCommonDataFormats
                                        O2::CommonUtils
                                        Boost::program_options)
