# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsCPV
               SOURCES src/CPVBlockHeader.cxx 
                       src/Digit.cxx 
                       src/Cluster.cxx 
                       src/TriggerRecord.cxx
		       src/CTF.cxx
                PUBLIC_LINK_LIBRARIES O2::CommonDataFormat 
                                     O2::Headers 
                                     O2::MathUtils 
                                     O2::DetectorsBase 
                                     O2::CPVBase 
                                     O2::SimulationDataFormat
                                     Boost::serialization)

o2_target_root_dictionary(DataFormatsCPV
                          HEADERS include/DataFormatsCPV/CPVBlockHeader.h
                                  include/DataFormatsCPV/Digit.h
                                  include/DataFormatsCPV/Cluster.h
                                  include/DataFormatsCPV/TriggerRecord.h
				  include/DataFormatsCPV/CTF.h)
