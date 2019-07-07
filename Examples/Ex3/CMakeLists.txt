o2_add_library(Ex3
               SOURCES src/A.cxx src/B.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)

o2_target_root_dictionary(Ex3 HEADERS include/Ex3/A.h src/B.h)

o2_add_executable(ex3
                  SOURCES src/run.cxx
                  PUBLIC_LINK_LIBRARIES O2::Ex3 O2::Ex2
                  COMPONENT_NAME example)
