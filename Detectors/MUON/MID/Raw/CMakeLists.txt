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
  MIDRaw
  SOURCES src/CrateMapper.cxx
          src/CRUBareDecoder.cxx
          src/CRUUserLogicDecoder.cxx
          src/CRUUserLogicEncoder.cxx
          src/Decoder.cxx
          src/ELinkDecoder.cxx
          src/Encoder.cxx
          src/LocalBoardRO.cxx
          src/RawBuffer.cxx
          src/RawFileReader.cxx
  PUBLIC_LINK_LIBRARIES ms_gsl::ms_gsl O2::DataFormatsMID O2::Headers
                        O2::CommonConstants O2::DetectorsRaw O2::MIDBase)

add_subdirectory(exe)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
