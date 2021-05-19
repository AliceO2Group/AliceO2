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
  ZDCRaw
  SOURCES src/DumpRaw.cxx
          src/raw-parser.cxx
          src/RawReaderZDC.cxx
  PUBLIC_LINK_LIBRARIES
    O2::SimulationDataFormat
    O2::ZDCBase
    O2::ZDCSimulation
    O2::DataFormatsZDC
    O2::CCDB
    O2::SimConfig
    O2::DPLUtils
    O2::DetectorsRaw
    O2::Headers)

o2_target_root_dictionary(ZDCRaw HEADERS include/ZDCRaw/DumpRaw.h)

o2_add_executable(
  raw-parser
  COMPONENT_NAME zdc
  SOURCES src/raw-parser.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DPLUtils
    O2::ZDCSimulation
    O2::ZDCRaw
    O2::DetectorsRaw
    O2::DetectorsCommonDataFormats
    O2::CommonUtils)
