o2_add_library(DataFormatsCTP
        SOURCES src/Digits.cxx
        PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
        O2::Headers
        O2::SimulationDataFormat
        )
o2_target_root_dictionary(DataFormatsCTP
HEADERS include/DataFormatsCTP/Digits.h
        )