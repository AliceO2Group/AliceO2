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

o2_add_library(MCHMappingSegContour
               SOURCES src/CathodeSegmentationContours.cxx
                       src/CathodeSegmentationSVGWriter.cxx
                       src/SegmentationContours.cxx
               PUBLIC_LINK_LIBRARIES O2::MCHMappingImpl4 O2::MCHContour
                                     Boost::boost)

o2_add_executable(mapping-svg-segmentation3
                  COMPONENT_NAME mch
                  SOURCES src/SVGSegmentation.cxx
                  PUBLIC_LINK_LIBRARIES O2::MCHMappingSegContour
                                        Boost::program_options)
