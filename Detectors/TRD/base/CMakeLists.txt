o2_add_library(TRDBase <<<<<<< 8701cd13c68d8fb591dbefe25e4c920c17c67879
               SOURCES src/TRDPadPlane.cxx
                       src/TRDGeometryBase.cxx
                       src/TRDGeometry.cxx
                       src/TRDGeometryFlat.cxx
                       src/TRDCommonParam.cxx
                       src/TRDSimParam.cxx
                       src/PadResponse.cxx
                       src/Digit.cxx
                       src/TRDArrayADC.cxx
                       src/TRDArrayDictionary.cxx
                       src/TRDArraySignal.cxx
                       src/TRDCalPadStatus.cxx
                       src/TRDCalSingleChamberStatus.cxx
                       src/CalDet.cxx
                       src/CalROC.cxx
                       src/TRDFeeParam.cxx
                       src/TRDSignalIndex.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonGPU
                                     O2::DetectorsCommonDataFormats O2::Field
                                     O2::DetectorsBase ROOT::Physics)

o2_add_test(DiffusionCoefficient SOURCE test/testTRDDiffusionCoefficient.cxx
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase)

o2_target_root_dictionary(TRDBase
                          HEADERS include/TRDBase/TRDPadPlane.h
                                  include/TRDBase/TRDGeometryBase.h
                                  include/TRDBase/TRDGeometry.h
                                  include/TRDBase/TRDGeometryFlat.h
                                  include/TRDBase/TRDSimParam.h
                                  include/TRDBase/TRDCommonParam.h
                                  include/TRDBase/PadResponse.h
                                  include/TRDBase/Digit.h
                                  include/TRDBase/DigitIndex.h
                                  include/TRDBase/TRDDigitsParam.h
                                  include/TRDBase/TRDArrayADC.h
                                  include/TRDBase/TRDArrayDictionary.h
                                  include/TRDBase/TRDArraySignal.h
                                  include/TRDBase/TRDCalPadStatus.h
                                  include/TRDBase/TRDCalSingleChamberStatus.h
                                  include/TRDBase/CalDet.h
                                  include/TRDBase/CalROC.h
                                  include/TRDBase/TRDFeeParam.h
                                  include/TRDBase/TRDSignalIndex.h)
