# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.


o2_add_library(EMCALReconstruction
               SOURCES src/RawReaderFile.cxx
                       src/RawBuffer.cxx
                       src/RAWDataHeader.cxx
                       src/AltroDecoder.cxx
                       src/Mapper.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::Headers
                                     AliceO2::InfoLogger
                                     O2::EMCALBase)

o2_target_root_dictionary(
                          EMCALReconstruction
                          HEADERS include/EMCALReconstruction/RawReaderFile.h
                                  include/EMCALReconstruction/AltroDecoder.h
                                  include/EMCALReconstruction/Mapper.h)

o2_add_executable(rawreader-file
                  COMPONENT_NAME emcal
                  PUBLIC_LINK_LIBRARIES O2::EMCALReconstruction
                  SOURCES run/rawReaderFile.cxx)

o2_data_file(COPY files DESTINATION Detectors/EMCAL)