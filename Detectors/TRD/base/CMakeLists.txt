# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(TRDBase
               SOURCES src/PadPlane.cxx
                       src/GeometryBase.cxx
                       src/Geometry.cxx
                       src/GeometryFlat.cxx
                       src/PadResponse.cxx
                       src/FeeParam.cxx
                       src/RecoParam.cxx
                       src/ChamberStatus.cxx
                       src/Calibrations.cxx
                       src/CalOnlineGainTables.cxx
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
                                  include/TRDBase/PadResponse.h
                                  include/TRDBase/FeeParam.h
                                  include/TRDBase/RecoParam.h
                                  include/TRDBase/Calibrations.h
                                  include/TRDBase/PadParameters.h
                                  include/TRDBase/PadCalibrations.h
                                  include/TRDBase/ChamberCalibrations.h
                                  include/TRDBase/PadCalibrationsAliases.h
                                  include/TRDBase/ChamberStatus.h
                                  include/TRDBase/CalOnlineGainTables.h
                                  include/TRDBase/TrackletTransformer.h)

o2_add_test(Geometry
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase
            SOURCES test/testTRDGeometry.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS trd
            )
o2_add_test(Transformations
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase O2::DataFormatsTRD
            SOURCES test/testCoordinateTransforms.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS trd
            )
