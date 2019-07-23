# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MFTTracking
               TARGETVARNAME targetName
               SOURCES src/Tracker.cxx
	               src/ROframe.cxx
	               src/Cluster.cxx
	               src/IOUtils.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonConstants
                                     O2::DataFormatsITSMFT
                                     O2::SimulationDataFormat
                                     O2::MFTBase
                                     O2::DataFormatsMFT)

o2_target_root_dictionary(MFTTracking
                          HEADERS include/MFTTracking/TrackCA.h
                          LINKDEF src/MFTTrackingLinkDef.h)
				

