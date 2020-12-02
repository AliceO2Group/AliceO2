#Copyright CERN and copyright holders of ALICE O2.This software is distributed
#under the terms of the GNU General Public License v3(GPL Version 3), copied
#verbatim in the file "COPYING".
#
#See http: //alice-o2.web.cern.ch/license for full licensing information.
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

o2_add_library(PHOSReconstruction
               SOURCES src/Clusterer.cxx
                       src/FullCluster.cxx
                       src/RawReaderMemory.cxx
                       src/RawBuffer.cxx
                       src/RawHeaderStream.cxx
                       src/RawPayload.cxx
                       src/AltroDecoder.cxx
                       src/Bunch.cxx
                       src/Channel.cxx
                       src/CaloRawFitter.cxx
               PUBLIC_LINK_LIBRARIES O2::PHOSBase
                                     O2::PHOSCalib
                                     O2::DataFormatsPHOS
                                     O2::DetectorsRaw
                                     AliceO2::InfoLogger)

o2_target_root_dictionary(PHOSReconstruction
                          HEADERS include/PHOSReconstruction/RawReaderMemory.h
                                  include/PHOSReconstruction/RawBuffer.h
                                  include/PHOSReconstruction/RawHeaderStream.h
                                  include/PHOSReconstruction/RawPayload.h
                                  include/PHOSReconstruction/RawReaderError.h
                                  include/PHOSReconstruction/Bunch.h
                                  include/PHOSReconstruction/RawDecodingError.h
                                  include/PHOSReconstruction/AltroDecoder.h
                                  include/PHOSReconstruction/Channel.h
                                  include/PHOSReconstruction/CaloRawFitter.h
                                  include/PHOSReconstruction/Clusterer.h 
                                  include/PHOSReconstruction/FullCluster.h)
o2_add_executable(rawreader-file
                  COMPONENT_NAME phos
                  PUBLIC_LINK_LIBRARIES O2::PHOSReconstruction
                  SOURCES run/rawReaderFile.cxx)

