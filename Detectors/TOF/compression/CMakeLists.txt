# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(TOFCompression
               SOURCES src/Compressor.cxx
               	       src/CompressorTask.cxx
               PUBLIC_LINK_LIBRARIES O2::TOFBase O2::Framework O2::Headers O2::DataFormatsTOF
	                             O2::DetectorsRaw
	       )

o2_add_executable(compressor
                  COMPONENT_NAME tof
                  SOURCES src/tof-compressor.cxx
                  PUBLIC_LINK_LIBRARIES O2::TOFCompression
		  TARGETVARNAME tofcompressor
		  )

o2_add_executable(compressed-inspector
                  COMPONENT_NAME tof
                  SOURCES src/tof-compressed-inspector.cxx
                  PUBLIC_LINK_LIBRARIES O2::TOFWorkflowUtils
		  )

o2_add_executable(compressed-analysis
                  COMPONENT_NAME tof
                  SOURCES src/tof-compressed-analysis.cxx
                  PUBLIC_LINK_LIBRARIES O2::TOFWorkflowUtils
		  )

if(NOT APPLE)

 set_property(TARGET ${tofcompressor} PROPERTY LINK_WHAT_YOU_USE ON)

endif()
