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

o2_add_library(G3Setup
               SOURCES src/G3Config.cxx
               PUBLIC_LINK_LIBRARIES MC::Geant3 FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

# Optional VecGeom navigation for Geant4 (G4.navmode=kVecGeom). It needs TGeo2VecGeom and a VecGeom
# with the BVH navigator of the VNavigator family (BVHNavigatorV). Linked PRIVATE: VecGeom types
# never appear in G4Setup's public headers.
find_package(TGeo2VecGeom CONFIG QUIET)
set(G4SETUP_WITH_VECGEOM OFF)
if(TGeo2VecGeom_FOUND)
  find_path(O2_VECGEOM_BVHNAVIGATORV_INCLUDE VecGeom/navigation/BVHNavigatorV.h
            HINTS ${VecGeom_INCLUDE_DIRS} ${VecGeom_DIR}/../../../include $ENV{VECGEOM_ROOT}/include)
  if(O2_VECGEOM_BVHNAVIGATORV_INCLUDE)
    set(G4SETUP_WITH_VECGEOM ON)
  endif()
endif()

set(G4SETUP_SOURCES src/G4Config.cxx src/G4RunConfiguration.cxx src/G4LocalFieldConstruction.cxx
                    src/VecGeomNavigation.cxx)
if(G4SETUP_WITH_VECGEOM)
  list(APPEND G4SETUP_SOURCES src/VecGeomG4Map.cxx src/VecGeomChecks.cxx src/VecGeomG4NavigatorBase.cxx
                              src/VecGeomG4Navigator.cxx
                              src/VecGeomG4PropagatingNavigator.cxx)
endif()

o2_add_library(G4Setup
               TARGETVARNAME targetG4Setup
               SOURCES ${G4SETUP_SOURCES}
               PUBLIC_LINK_LIBRARIES MC::Geant4VMC MC::Geant4 FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup O2::FastSim O2::DetectorsBase
)

if(G4SETUP_WITH_VECGEOM)
  target_compile_definitions(${targetG4Setup} PRIVATE O2_WITH_VECGEOM)
  target_link_libraries(${targetG4Setup} PRIVATE TGeo2VecGeom::TGeo2VecGeom)
else()
  message(STATUS "G4.navmode=kVecGeom not built: it needs TGeo2VecGeom and a VecGeom with BVHNavigatorV")
endif()

o2_add_library(FLUKASetup
               SOURCES src/FlukaConfig.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(MCReplaySetup
               SOURCES src/MCReplayConfig.cxx
               PUBLIC_LINK_LIBRARIES MC::MCReplay FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(O2TrivialMCEngineSetup
               SOURCES src/O2TrivialMCConfig.cxx
               PUBLIC_LINK_LIBRARIES O2::O2TrivialMC FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(SimSetup
               SOURCES  src/SimSetup.cxx src/SetCuts.cxx src/FlukaParam.cxx src/MCReplayParam.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonUtils O2::DetectorsBase O2::SimConfig
)

o2_target_root_dictionary(SimSetup
                          HEADERS include/SimSetup/SimSetup.h
                                  include/SimSetup/FlukaParam.h
                                  include/SimSetup/MCReplayParam.h
                          LINKDEF src/GConfLinkDef.h)

o2_add_test_root_macro(UserDecay.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)

o2_add_test_root_macro(commonConfig.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)

o2_add_test_root_macro(g3libs.C LABELS simsetup)

o2_add_test_root_macro(g3Config.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)

#o2_add_test_root_macro(g4Config.C
#                       PUBLIC_LINK_LIBRARIES O2::SimSetup
#                       LABELS simsetup)


o2_data_file(COPY data  DESTINATION Detectors/gconfig/)
install(FILES src/StandardSteppingTrackRefHook.macro src/FlukaRuntimeConfig.macro
              src/KeepStepCylinders.macro DESTINATION share/Detectors/gconfig/)
