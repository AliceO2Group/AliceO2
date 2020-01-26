# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DetectorsRaw
               SOURCES src/RawFileReader.cxx
	               src/HBFUtils.cxx  
               PUBLIC_LINK_LIBRARIES FairRoot::Base
		                     O2::Headers
		                     O2::CommonDataFormat  
                                     O2::DetectorsCommonDataFormats)

o2_target_root_dictionary(DetectorsRaw
                          HEADERS include/DetectorsRaw/RawFileReader.h
				  include/DetectorsRaw/HBFUtils.h)


			
o2_add_executable(filecheck 
                  COMPONENT_NAME Raw
                  SOURCES src/rawfileCheck.cxx
                  PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
                                        Boost::program_options)

o2_add_test(HBFUtils
            PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
                                  O2::Steer
            SOURCES test/testHBFUtils.cxx
            COMPONENT_NAME raw
            LABELS raw)
