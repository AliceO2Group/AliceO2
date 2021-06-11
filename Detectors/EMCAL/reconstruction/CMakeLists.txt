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
               SOURCES src/RawReaderMemory.cxx
                       src/RawBuffer.cxx
                       src/RawHeaderStream.cxx
                       src/RawPayload.cxx
                       src/AltroDecoder.cxx
                       src/Bunch.cxx
                       src/Channel.cxx
                       src/CaloFitResults.cxx
                       src/CaloRawFitter.cxx
                       src/CaloRawFitterStandard.cxx
                       src/CaloRawFitterGamma2.cxx
                       src/ClusterizerParameters.cxx
                       src/Clusterizer.cxx
                       src/ClusterizerTask.cxx
                       src/DigitReader.cxx
                       src/CTFCoder.cxx
                       src/CTFHelper.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::Headers
                                     AliceO2::InfoLogger
                                     O2::DataFormatsEMCAL
                                     O2::DetectorsBase
                                     O2::DetectorsRaw
                                     O2::EMCALBase
                                     O2::rANS
                                     Microsoft.GSL::GSL)

o2_target_root_dictionary(
                          EMCALReconstruction
                          HEADERS include/EMCALReconstruction/RawReaderMemory.h
                                  include/EMCALReconstruction/AltroDecoder.h
                                  include/EMCALReconstruction/RawPayload.h
                                  include/EMCALReconstruction/Bunch.h
                                  include/EMCALReconstruction/Channel.h
                                  include/EMCALReconstruction/CaloFitResults.h
                                  include/EMCALReconstruction/CaloRawFitter.h
                                  include/EMCALReconstruction/CaloRawFitterStandard.h
                                  include/EMCALReconstruction/CaloRawFitterGamma2.h
                                  include/EMCALReconstruction/ClusterizerParameters.h
                                  include/EMCALReconstruction/Clusterizer.h
                                  include/EMCALReconstruction/ClusterizerTask.h
                                  include/EMCALReconstruction/DigitReader.h
)

o2_add_executable(rawreader-file
                  COMPONENT_NAME emcal
                  PUBLIC_LINK_LIBRARIES O2::EMCALReconstruction
                  SOURCES run/rawReaderFile.cxx)

o2_add_test_root_macro(macros/RawFitterTESTs.C
            PUBLIC_LINK_LIBRARIES O2::EMCALReconstruction O2::Headers
            LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/RawFitterTESTMulti.C
            PUBLIC_LINK_LIBRARIES O2::EMCALReconstruction O2::Headers
            LABELS emcal COMPILE_ONLY)

