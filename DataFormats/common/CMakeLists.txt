# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CommonDataFormat
               SOURCES src/InteractionRecord.cxx
                       src/BunchFilling.cxx
                       src/FlatHisto1D.cxx
                       src/FlatHisto2D.cxx
                       src/AbstractRefAccessor.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonConstants O2::GPUCommon
                                     ROOT::Core FairRoot::Base ms_gsl::ms_gsl)

o2_target_root_dictionary(CommonDataFormat
                          HEADERS include/CommonDataFormat/TimeStamp.h
                                  include/CommonDataFormat/EvIndex.h
                                  include/CommonDataFormat/RangeReference.h
                                  include/CommonDataFormat/InteractionRecord.h
                                  include/CommonDataFormat/BunchFilling.h
                                  include/CommonDataFormat/AbstractRef.h
                                  include/CommonDataFormat/FlatHisto1D.h
                                  include/CommonDataFormat/FlatHisto2D.h)                                

o2_add_test(TimeStamp
            SOURCES test/testTimeStamp.cxx
            COMPONENT_NAME CommonDataFormat
            PUBLIC_LINK_LIBRARIES O2::CommonDataFormat)

o2_add_test(RangeRef
            SOURCES test/testRangeRef.cxx
            COMPONENT_NAME CommonDataFormat
            PUBLIC_LINK_LIBRARIES O2::CommonDataFormat)

o2_add_test(FlatHisto
            SOURCES test/testFlatHisto.cxx
            COMPONENT_NAME CommonDataFormat
            PUBLIC_LINK_LIBRARIES O2::CommonDataFormat)

o2_add_test(AbstractRefAccessor
            SOURCES test/testAbstractRefAccessor.cxx
            COMPONENT_NAME CommonDataFormat
            PUBLIC_LINK_LIBRARIES O2::CommonDataFormat)
