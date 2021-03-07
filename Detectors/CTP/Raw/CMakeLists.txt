o2_add_library(CTPRaw
            PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
                                  O2::Steer
                                  O2::CommonUtils
                                  O2::DataFormatsCTP
            SOURCES src/CTPRawWriter.cxx
                    src/CTPRawReader.cxx
)
o2_target_root_dictionary(CTPRaw
                                      HEADERS include/CTPRaw/CTPRawWriter.h
                                              include/CTPRaw/CTPRawReader.h
            )
o2_add_test_root_macro(macro/writeReadCTPRaw.C
            PUBLIC_LINK_LIBRARIES O2::CTPRaw
            O2::CommonUtils
            LABELS raw)
