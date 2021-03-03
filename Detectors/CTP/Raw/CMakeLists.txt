o2_add_library(CTPRaw
            PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
                                  O2::Steer
	                          O2::DPLUtils
            #SOURCES test/CTPtestRawReaderWriter.cxx
)
o2_target_root_dictionary(CTPRaw
                                      HEADERS include/CTPRaw/CTPRawWriterDummy.h
            )
