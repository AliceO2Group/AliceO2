set(BUCKET_NAME mch_contour_bucket)
set(MODULE_NAME MCHContour)

add_library(MCHContour INTERFACE)

install(DIRECTORY include/MCHContour DESTINATION include)

O2_GENERATE_TESTS(

        BUCKET_NAME mch_contour_bucket

        TEST_SRCS
        test/BBox.cxx
        test/Contour.cxx
        test/ContourCreator.cxx
        test/Edge.cxx
        test/Interval.cxx
        test/Polygon.cxx
        test/SegmentTree.cxx
        test/Vertex.cxx

)

# custom target so that some IDE (e.g. CLion) recognize those files as
# being part of the project even though they are not part of a CMake target
add_custom_target(MCHContour_ide SOURCES
        include/MCHContour/BBox.h
        include/MCHContour/Contour.h
        include/MCHContour/ContourCreator.h
        include/MCHContour/ContourCreator.inl
        include/MCHContour/Edge.h
        include/MCHContour/Helper.h
        include/MCHContour/Interval.h
        include/MCHContour/Polygon.h
        include/MCHContour/SegmentTree.h
        include/MCHContour/SVGWriter.h
        include/MCHContour/Vertex.h
        )
