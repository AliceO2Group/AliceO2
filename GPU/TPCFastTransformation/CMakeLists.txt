o2_add_library(TPCFastTransformation
               SOURCES IrregularSpline1D.cxx IrregularSpline2D3D.cxx
                       IrregularSpline2D3DCalibrator.cxx TPCDistortionIRS.cxx
                       TPCFastTransform.cxx
               PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}
               PUBLIC_LINK_LIBRARIES O2::CommonGPU Vc::Vc ROOT::Core)

o2_target_root_dictionary(TPCFastTransformation
                          HEADERS IrregularSpline1D.h IrregularSpline2D3D.h
                                  IrregularSpline2D3DCalibrator.h
                                  TPCFastTransform.h TPCDistortionIRS.h
                          LINKDEF TPCFastTransformationLinkDef_O2.h)

