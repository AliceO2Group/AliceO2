o2_add_header_only_library(Algorithm INTERFACE_LINK_LIBRARIES O2::Headers)

o2_add_test(o2formatparser
            SOURCES
            test/o2formatparser.cxx
            COMPONENT_NAME
            Algorithm
            PUBLIC_LINK_LIBRARIES
            Algorithm)

o2_add_test(headerstack
            SOURCES
            test/headerstack.cxx
            COMPONENT_NAME
            Algorithm
            PUBLIC_LINK_LIBRARIES
            Algorithm)

o2_add_test(parser
            SOURCES
            test/parser.cxx
            COMPONENT_NAME
            Algorithm
            PUBLIC_LINK_LIBRARIES
            Algorithm)

o2_add_test(tableview
            SOURCES
            test/tableview.cxx
            COMPONENT_NAME
            Algorithm
            PUBLIC_LINK_LIBRARIES
            Algorithm)

o2_add_test(pageparser
            SOURCES
            test/pageparser.cxx
            COMPONENT_NAME
            Algorithm
            PUBLIC_LINK_LIBRARIES
            Algorithm)

o2_add_test(mpl_tools SOURCES test/test_mpl_tools.cxx COMPONENT_NAME Algorithm)

o2_add_test(RangeTokenizer SOURCES test/test_RangeTokenizer.cxx COMPONENT_NAME
            Algorithm)

# FIXME: reconnect man pages generation
#
# * O2_GENERATE_MAN(NAME Algorithm SECTION 3)
# * O2_GENERATE_MAN(NAME algorithm_parser SECTION 3)
