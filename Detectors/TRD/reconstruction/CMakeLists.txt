#Copyright CERN and copyright holders of ALICE O2.This software is distributed
#under the terms of the GNU General Public License v3(GPL Version 3), copied
#verbatim in the file "COPYING".
#
#See http: //alice-o2.web.cern.ch/license for full licensing information.
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

o2_add_library(TRDReconstruction
               SOURCES src/CTFCoder.cxx
                       src/CTFHelper.cxx
                       src/DigitsParser.cxx
                       src/TrackletsParser.cxx
                       src/CruRawReader.cxx
                       src/CompressedRawReader.cxx
                       src/EpnDataReaderTask.cxx
                       src/CruCompressorTask.cxx
               PUBLIC_LINK_LIBRARIES O2::TRDBase O2::DataFormatsTRD O2::DetectorsRaw O2::rANS ms_gsl::ms_gsl)

                   # o2_target_root_dictionary(TRDReconstruction
                   #                           HEADERS )


o2_add_executable(crucompressor
    COMPONENT_NAME trd
    SOURCES src/CruCompressor.cxx
    PUBLIC_LINK_LIBRARIES O2::TRDReconstruction
    )

o2_add_executable(epndatareader
    COMPONENT_NAME trd
    SOURCES src/EpnDataReader.cxx
    PUBLIC_LINK_LIBRARIES O2::TRDReconstruction
    )
