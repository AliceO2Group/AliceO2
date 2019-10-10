
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TRDBase
               SOURCES src/TRDPadPlane.cxx
                       src/TRDGeometryBase.cxx
                       src/TRDGeometry.cxx
                       src/TRDGeometryFlat.cxx
                       src/TRDCommonParam.cxx
                       src/TRDSimParam.cxx
                       src/PadResponse.cxx
                       src/Digit.cxx
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
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                     O2::DetectorsCommonDataFormats
                                     O2::Field
                                     O2::DetectorsBase
                                     ROOT::Physics
                                     O2::SimulationDataFormat
                                     O2::CCDB)

o2_target_root_dictionary(TRDBase
                          HEADERS include/TRDBase/TRDPadPlane.h
                                  include/TRDBase/TRDGeometryBase.h
                                  include/TRDBase/TRDGeometry.h
                                  include/TRDBase/TRDGeometryFlat.h
                                  include/TRDBase/TRDSimParam.h
                                  include/TRDBase/TRDCommonParam.h
                                  include/TRDBase/PadResponse.h
                                  include/TRDBase/Digit.h
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
                                  include/TRDBase/CalOnlineGainTables.h)

o2_add_test(DiffusionCoefficient
            SOURCES test/testTRDDiffusionCoefficient.cxx
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase
            ENVIRONMENT VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/share
            LABELS trd)
