
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TRDBase
               SOURCES src/PadPlane.cxx
                       src/GeometryBase.cxx
                       src/Geometry.cxx
                       src/GeometryFlat.cxx
                       src/CommonParam.cxx
                       src/Digit.cxx
                       src/DiffAndTimeStructEstimator.cxx
                       src/SimParam.cxx
                       src/PadResponse.cxx
                       src/CalDet.cxx
                       src/CalROC.cxx
                       src/FeeParam.cxx
                       src/PadStatus.cxx
                       src/LocalVDrift.cxx
                       src/LocalT0.cxx
                       src/LocalGainFactor.cxx
                       src/ChamberStatus.cxx
                       src/Calibrations.cxx
                       src/ChamberNoise.cxx
                       src/CalOnlineGainTables.cxx
                       src/Tracklet.cxx
                       src/TrackletTransformer.cxx
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                     O2::GPUUtils
                                     O2::DetectorsCommonDataFormats
                                     O2::Field
                                     O2::DetectorsBase
                                     ROOT::Physics
                                     O2::SimulationDataFormat
                                     O2::DataFormatsTRD
                                     O2::CCDB)

o2_target_root_dictionary(TRDBase
                          HEADERS include/TRDBase/PadPlane.h
                                  include/TRDBase/GeometryBase.h
                                  include/TRDBase/Geometry.h
                                  include/TRDBase/GeometryFlat.h
                                  include/TRDBase/SimParam.h
                                  include/TRDBase/CommonParam.h
                                  include/TRDBase/Digit.h
                                  include/TRDBase/PadResponse.h
                                  include/TRDBase/MCLabel.h
                                  include/TRDBase/CalDet.h
                                  include/TRDBase/CalROC.h
                                  include/TRDBase/FeeParam.h
                                  include/TRDBase/PadParameters.h
                                  include/TRDBase/PadCalibrations.h
                                  include/TRDBase/ChamberCalibrations.h
                                  include/TRDBase/PadStatus.h
                                  include/TRDBase/LocalVDrift.h
                                  include/TRDBase/LocalT0.h
                                  include/TRDBase/LocalGainFactor.h
                                  include/TRDBase/ChamberStatus.h
                                  include/TRDBase/Calibrations.h
                                  include/TRDBase/ChamberNoise.h
                                  include/TRDBase/CalOnlineGainTables.h
                                  include/TRDBase/Tracklet.h
                                  include/TRDBase/TrackletTransformer.h)

o2_add_test(DiffusionCoefficient
            SOURCES test/testTRDDiffusionCoefficient.cxx
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase
            ENVIRONMENT VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/share
            LABELS trd)

o2_add_test(Geometry
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase
            SOURCES test/testTRDGeometry.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS trd
            )
o2_add_test(RawData
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase O2::DataFormatsTRD
            SOURCES test/testRawData.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS trd
            )
