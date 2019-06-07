o2_add_library(Field
        SOURCES
        src/MagFieldContFact.cxx
        src/MagFieldFact.cxx
        src/MagFieldFast.cxx
        src/MagFieldParam.cxx
        src/MagneticField.cxx
        src/MagneticWrapperChebyshev.cxx
        PUBLIC_LINK_LIBRARIES
        O2::MathUtils)

o2_target_root_dictionary(Field
        HEADERS
        include/Field/MagneticWrapperChebyshev.h
        include/Field/MagneticField.h
        include/Field/MagFieldParam.h
        include/Field/MagFieldContFact.h
        include/Field/MagFieldFast.h
        include/Field/MagFieldFact.h)

o2_add_test(MagneticField
        SOURCES
        test/testMagneticField.cxx
        PUBLIC_LINK_LIBRARIES
        Field
        COMPONENT_NAME
        Field
        COMMAND_LINE_ARGS --file ${CMAKE_CURRENT_LIST_DIR}/../maps/mfchebKGI_sym.root)

o2_add_test_root_macro(macro/extractMapsAsText.C PUBLIC_LINK_LIBRARIES Field)
o2_add_test_root_macro(macro/createMapsFromText.C PUBLIC_LINK_LIBRARIES Field)
