o2_add_library(DetectorsBase
               SOURCES src/Detector.cxx
                       src/GeometryManager.cxx
                       src/MaterialManager.cxx
                       src/Propagator.cxx
                       src/MatLayerCyl.cxx
                       src/MatLayerCylSet.cxx
                       src/Ray.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base
                                     O2::CommonUtils
                                     O2::DetectorsCommonDataFormats
                                     O2::CommonGPU
                                     O2::ReconstructionDataFormats
                                     O2::Field
                                     FairMQ::FairMQ
                                     O2::DataFormatsParameters
                                     ROOT::VMC)

o2_target_root_dictionary(DetectorsBase
                          HEADERS include/DetectorsBase/Detector.h
                                  include/DetectorsBase/GeometryManager.h
                                  include/DetectorsBase/MaterialManager.h
                                  include/DetectorsBase/Propagator.h
                                  include/DetectorsBase/Triggers.h
                                  include/DetectorsBase/Ray.h
                                  include/DetectorsBase/MatCell.h
                                  include/DetectorsBase/MatLayerCyl.h
                                  include/DetectorsBase/MatLayerCylSet.h)

if(BUILD_SIMULATION)
  o2_add_test(MadBudLUT
              SOURCES test/testMatBudLUT.cxx
              COMPONENT_NAME DetectorsBase
              PUBLIC_LINK_LIBRARIES O2::DetectorsBase)
endif()

o2_add_test_root_macro(test/buildMatBudLUT.C
                       PUBLIC_LINK_LIBRARIES DetectorsBase)
