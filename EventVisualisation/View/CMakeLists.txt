o2_add_library(EventVisualisationView
               SOURCES src/MultiView.cxx src/Initializer.cxx
               PUBLIC_LINK_LIBRARIES O2::EventVisualisationBase
                                     O2::EventVisualisationDetectors)

o2_target_root_dictionary(EventVisualisationView
                          HEADERS include/EventVisualisationView/MultiView.h
                                  include/EventVisualisationView/Initializer.h)

o2_add_executable(eve
                  SOURCES src/main.cxx
                  PUBLIC_LINK_LIBRARIES O2::EventVisualisationView)
