o2_add_library(CTPSimulation
        SOURCES src/Digitizer.cxx
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::Framework
        O2::DataFormatsCTP
        O2::Headers)
o2_target_root_dictionary(CTPSimulation HEADERS
        include/CTPSimulation/Digitizer.h
        )