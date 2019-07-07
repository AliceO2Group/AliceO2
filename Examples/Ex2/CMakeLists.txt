o2_add_library(Ex2
               SOURCES src/A.cxx src/B.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)

o2_target_root_dictionary(Ex2
                          HEADERS include/Ex2/A.h src/B.h
                          LINKDEF src/Ex2LinkDef.h)
