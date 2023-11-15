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

o2_add_library(TPCSpaceCharge
               TARGETVARNAME targetName
               SOURCES src/SpaceCharge.cxx
                       src/PoissonSolver.cxx
                       src/TriCubic.cxx
                       src/DataContainer3D.cxx
               PUBLIC_LINK_LIBRARIES O2::TPCBase
                                     O2::Field
                                     Vc::Vc
                                     ROOT::Core
                                     ROOT::ROOTDataFrame
                                     O2::DataFormatsParameters)


o2_target_root_dictionary(TPCSpaceCharge
                          HEADERS include/TPCSpaceCharge/PoissonSolver.h
                                  include/TPCSpaceCharge/SpaceCharge.h
                                  include/TPCSpaceCharge/RegularGrid3D.h
                                  include/TPCSpaceCharge/DataContainer3D.h
                                  include/TPCSpaceCharge/PoissonSolverHelpers.h
                                  include/TPCSpaceCharge/SpaceChargeHelpers.h
                                  include/TPCSpaceCharge/TriCubic.h
                                  include/TPCSpaceCharge/Vector.h
                                  include/TPCSpaceCharge/Vector3D.h
                                  include/TPCSpaceCharge/SpaceChargeParameter.h
                          LINKDEF src/TPCSpacechargeLinkDef.h)

o2_add_test_root_macro(macro/calculateDistortionsCorrections.C
                       PUBLIC_LINK_LIBRARIES O2::TPCSpaceCharge
                       LABELS tpc COMPILE_ONLY)

o2_add_test_root_macro(macro/createResidualDistortionObject.C
                      PUBLIC_LINK_LIBRARIES O2::TPCSpaceCharge
                                            O2::CommonUtils
                      LABELS tpc)


install(FILES macro/createSCHistosFromHits.C
        DESTINATION share/macro/)

o2_add_test_root_macro(macro/createSCHistosFromHits.C
                      PUBLIC_LINK_LIBRARIES O2::TPCSpaceCharge
                                            O2::CommonUtils
                                            O2::TPCBase
                                            O2::TPCSimulation
                                            O2::DataFormatsTPC
                                            O2::MathUtils
                      LABELS tpc COMPILE_ONLY)

o2_add_test(PoissonSolver
            COMPONENT_NAME spacecharge
            PUBLIC_LINK_LIBRARIES O2::TPCSpaceCharge
            SOURCES test/testO2TPCPoissonSolver.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS tpc
            CONFIGURATIONS RelWithDebInfo Release MinRelSize)

o2_add_test(TriCubic
            COMPONENT_NAME spacecharge
            PUBLIC_LINK_LIBRARIES O2::TPCSpaceCharge
            SOURCES test/testO2TPCTricubic.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS tpc
            CONFIGURATIONS RelWithDebInfo Release MinRelSize)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
