# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_header_only_library(MCHContour)

o2_add_test(BBox
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/BBox.cxx)

o2_add_test(Contour
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/Contour.cxx)

o2_add_test(ContourCreator
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/ContourCreator.cxx)

o2_add_test(Edge
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/Edge.cxx)

o2_add_test(Interval
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/Interval.cxx)

o2_add_test(Polygon
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/Polygon.cxx)

o2_add_test(SegmentTree
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/SegmentTree.cxx)

o2_add_test(Vertex
            COMPONENT_NAME mchcontour
            LABELS mch muon
            PUBLIC_LINK_LIBRARIES O2::MCHContour
            SOURCES test/Vertex.cxx)
