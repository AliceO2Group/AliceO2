o2_add_test(Mapping
            SOURCES
            src/testMapping.cxx
            COMPONENT_NAME
            mid
            PUBLIC_LINK_LIBRARIES
            O2::MIDBase
            LABELS
            mid)

o2_add_test(Positions
            SOURCES
            src/Positions.cxx
            COMPONENT_NAME
            mid
            PUBLIC_LINK_LIBRARIES
            RapidJSON::RapidJSON
            O2::MIDBase
            LABELS
            mid)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test_random_pos.json
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
