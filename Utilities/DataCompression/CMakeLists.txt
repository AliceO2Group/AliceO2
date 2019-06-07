o2_add_test(dc_primitives SOURCES test/test_dc_primitives.cxx COMPONENT_NAME
        DataCompression)

o2_add_test(Fifo SOURCES test/test_Fifo.cxx COMPONENT_NAME DataCompression
        PUBLIC_LINK_LIBRARIES
        Threads::Threads)

o2_add_test(DataGenerator SOURCES test/test_DataGenerator.cxx COMPONENT_NAME
        DataCompression)

o2_add_test(HuffmanCodec
        SOURCES
        test/test_HuffmanCodec.cxx
        COMPONENT_NAME
        DataCompression
        PUBLIC_LINK_LIBRARIES
        O2::CommonUtils
        Boost::filesystem)

o2_add_test(DataDeflater SOURCES test/test_DataDeflater.cxx COMPONENT_NAME
        DataCompression)
