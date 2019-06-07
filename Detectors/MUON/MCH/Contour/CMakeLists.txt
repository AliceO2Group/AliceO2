o2_add_header_only_library(MCHContour)

o2_add_test(BBox
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/BBox.cxx)

o2_add_test(Contour
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/Contour.cxx)

o2_add_test(ContourCreator
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/ContourCreator.cxx)

o2_add_test(Edge
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/Edge.cxx)

o2_add_test(Interval
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/Interval.cxx)

o2_add_test(Polygon
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/Polygon.cxx)

o2_add_test(SegmentTree
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/SegmentTree.cxx)

o2_add_test(Vertex
            COMPONENT_NAME
            mchcontour
            LABELS
            mch
            PUBLIC_LINK_LIBRARIES
            MCHContour
            SOURCES
            test/Vertex.cxx
            PUBLIC_LINK_LIBRARIES
            MCHContour)
