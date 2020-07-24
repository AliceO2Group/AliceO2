# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TRDRaw
    SOURCES src/Cru2Tracklet.cxx
            src/Cru2TrackletTask.cxx
            src/Cru2TrackletTranslator.cxx
               PUBLIC_LINK_LIBRARIES O2::TRDBase O2::Framework O2::Headers 
                                     O2::DataFormatsTRD O2::DetectorsRaw
	       )

       o2_add_executable(cru2tracklet
                  COMPONENT_NAME trd
                  SOURCES src/Cru2Tracklet.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDRaw
		  )

