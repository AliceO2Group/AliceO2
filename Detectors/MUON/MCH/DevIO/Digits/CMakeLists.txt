# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHDigitIO
            SOURCES DigitWriter.cxx DigitReader.cxx DigitFileFormat.cxx
                    DigitReaderImpl.cxx DigitWriterImpl.cxx
                    DigitIOV0.cxx DigitIOV1.cxx
                    DigitIOV2.cxx DigitIOV3.cxx
                    IO.cxx
            PUBLIC_LINK_LIBRARIES Microsoft.GSL::GSL fmt::fmt O2::DataFormatsMCH)

o2_add_test(digits-io
            SOURCES testDigitIO.cxx
            COMPONENT_NAME mch
            LABELS muon;mch
            PUBLIC_LINK_LIBRARIES O2::MCHDigitIO)

o2_add_test(digits-io-v0
            SOURCES testDigitIOV0.cxx TestFileV0.cxx
            COMPONENT_NAME mch
            LABELS muon;mch
            PUBLIC_LINK_LIBRARIES O2::MCHDigitIO)

o2_add_executable(digits-r23-workflow
                  SOURCES digits-r23-workflow.cxx
                  COMPONENT_NAME mch
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsMCH
                                        O2::MCHBase O2::MCHMappingImpl3)

add_library(MCHDigitIOBaseTask OBJECT)
add_library(O2::MCHDigitIOBaseTask ALIAS MCHDigitIOBaseTask)
target_link_libraries(MCHDigitIOBaseTask PUBLIC O2::DataFormatsMCH O2::Framework)

target_sources(MCHDigitIOBaseTask PRIVATE DigitIOBaseTask.cxx)

o2_add_executable(
        digits-writer-workflow
        SOURCES digits-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::DPLUtils Boost::program_options
                              O2::MCHBase O2::MCHRawDecoder
                              O2::MCHDigitIO O2::MCHDigitIOBaseTask)

o2_add_executable(
        digits-file-reader-workflow
        SOURCES digits-file-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHBase O2::MCHDigitIO O2::MCHDigitIOBaseTask)

o2_add_executable(
        digits-random-generator-workflow
        SOURCES digits-random-generator-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHBase O2::MCHMappingImpl4 O2::MCHDigitIO O2::MCHDigitIOBaseTask)

o2_add_executable(
        digits-file-dumper
        SOURCES digits-file-dumper.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHDigitIO O2::MCHMappingImpl3 Boost::program_options)

