# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
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
                       src/TRDCalPadStatus.cxx
                       src/TRDCalSingleChamberStatus.cxx
                       src/CalDet.cxx
                       src/CalROC.cxx
                       src/TRDFeeParam.cxx
                       src/LTUParam.cxx
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                     O2::DetectorsCommonDataFormats
                                     O2::Field
                                     O2::DetectorsBase
                                     ROOT::Physics
                                     O2::SimulationDataFormat)

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
                                  include/TRDBase/TRDFeeParam.h
                                  include/TRDBase/LTUParam.h)

o2_add_test(DiffusionCoefficient
            SOURCES test/testTRDDiffusionCoefficient.cxx
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::TRDBase
            ENVIRONMENT VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/share
            LABELS trd)
