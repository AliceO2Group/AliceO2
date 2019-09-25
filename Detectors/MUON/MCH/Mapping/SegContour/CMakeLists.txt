# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHMappingSegContour3
               SOURCES src/CathodeSegmentationContours.cxx
                       src/CathodeSegmentationSVGWriter.cxx
                       src/SegmentationContours.cxx
               PUBLIC_LINK_LIBRARIES O2::MCHMappingImpl3 O2::MCHContour
                                     Boost::boost)

o2_add_executable(mapping-svg-segmentation3
                  COMPONENT_NAME mch
                  SOURCES src/SVGSegmentation.cxx
                  PUBLIC_LINK_LIBRARIES O2::MCHMappingSegContour3
                                        Boost::program_options)
