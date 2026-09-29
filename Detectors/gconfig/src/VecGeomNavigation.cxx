// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimSetup/VecGeomNavigation.h"

#include <fairlogger/Logger.h>
#include <sstream>
#include <string>

#ifdef O2_WITH_VECGEOM

// The O2 headers come first on purpose: VecGeom's build interface defines a VECGEOM macro,
// which would otherwise eat the VECGEOM enumerator of MatbudGeomBackend.
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/GeometryManagerParam.h"
#include "SimConfig/G4Params.h"

#include "VecGeomChecks.h"
#include "VecGeomG4Map.h"
#include "VecGeomG4Navigator.h"

#include "TG4RootDetectorConstruction.h"
#include "TG4RootNavMgr.h"

#include "G4EventManager.hh"
#include "G4FieldManager.hh"
#include "G4PropagatorInField.hh"
#include "G4SteppingManager.hh"
#include "G4TrackingManager.hh"
#include "G4TransportationManager.hh"

#include "TMCManager.h"
#include "TStopwatch.h"

#endif

namespace o2::simsetup
{

#ifdef O2_WITH_VECGEOM

bool isVecGeomNavigationAvailable() { return true; }

void installVecGeomNavigator()
{
  auto const& g4Params = o2::conf::G4Params::Instance();

  if (TMCManager::Instance() != nullptr) {
    LOG(fatal) << "G4.navmode=kVecGeom cannot be used with the multi-engine TMCManager: restoring a "
                  "geometry state across engines goes through the TGeo navigator";
  }
  if (o2::GeometryManagerParam::Instance().useParallelWorld) {
    LOG(fatal) << "G4.navmode=kVecGeom cannot be used with GeometryManagerParam.useParallelWorld: VecGeom "
                  "has no equivalent of the TGeo priority world";
  }
  auto* navMgr = TG4RootNavMgr::GetInstance();
  if (navMgr == nullptr || navMgr->GetDetConstruction() == nullptr) {
    LOG(fatal) << "G4.navmode=kVecGeom needs the Geant4 geometry built from TGeo by g4root, which is what "
                  "the geomRoot option provides; no TG4RootNavMgr was found";
  }
  auto* detConstruction = navMgr->GetDetConstruction();
  if (!detConstruction->IsConstructed()) {
    LOG(fatal) << "The Geant4 geometry has not been built yet; installVecGeomNavigator must be called after "
                  "the TGeant4 engine has been created";
  }

  if (!g4Params.vecgeomFlattenAssemblies) {
    LOG(fatal) << "G4.vecgeomNavigator=kStrict needs G4.vecgeomFlattenAssemblies=true: it enters a daughter by "
                  "locating inside it, which an assembly cannot answer";
  }

  TStopwatch timer;
  timer.Start();
  o2::base::GeometryManager::buildVecGeomGeometry(g4Params.vecgeomFlattenAssemblies);
  timer.Stop();
  LOG(info) << "VecGeom geometry built in " << timer.RealTime() << " s";

  timer.Start();
  // Owned here for the lifetime of the process; the navigator keeps a reference to it.
  static VecGeomG4Map map;
  map.build(*detConstruction, g4Params.vecgeomFlattenAssemblies);
  timer.Stop();
  LOG(info) << "VecGeom to Geant4 map built in " << timer.RealTime() << " s";

  if (!g4Params.vecgeomCheckVolumes.empty()) {
    std::stringstream names(g4Params.vecgeomCheckVolumes);
    std::string one;
    while (std::getline(names, one, ',')) {
      checkVecGeomVolume(one.c_str(), 20, 200);
    }
  }
  if (g4Params.vecgeomCheckRays > 0) {
    checkVecGeomRays(static_cast<std::size_t>(g4Params.vecgeomCheckRays));
  }
  if (g4Params.vecgeomCheckLocation > 0) {
    checkVecGeomLocation(static_cast<std::size_t>(g4Params.vecgeomCheckLocation));
  }

  auto* navigator = new VecGeomG4Navigator(map, g4Params.vecgeomPushDepth, g4Params.vecgeomZeroSafety);
  navigator->SetWorldVolume(detConstruction->GetTopPV());

  // Same sequence TG4RootNavMgr::SetNavigator uses, run here because by the time the engine
  // exists the navigator manager considers itself connected and refuses to swap.
  auto* trMgr = G4TransportationManager::GetTransportationManager();
  trMgr->SetNavigatorForTracking(navigator);
  auto* fieldMgr = trMgr->GetPropagatorInField()->GetCurrentFieldManager();
  delete trMgr->GetPropagatorInField();
  trMgr->SetPropagatorInField(new G4PropagatorInField(navigator, fieldMgr));
  trMgr->ActivateNavigator(navigator);
  if (auto* evtMgr = G4EventManager::GetEventManager()) {
    evtMgr->GetTrackingManager()->GetSteppingManager()->SetNavigator(navigator);
  }

  LOG(info) << "VecGeom navigator registered with the Geant4 transportation manager";
}

#else

bool isVecGeomNavigationAvailable() { return false; }

void installVecGeomNavigator()
{
  LOG(fatal) << "G4.navmode=kVecGeom needs O2 built against TGeo2VecGeom and a VecGeom with the BVH navigator of "
                "the VNavigator family (BVHNavigatorV), which were not found at configure time";
}

#endif

} // namespace o2::simsetup
