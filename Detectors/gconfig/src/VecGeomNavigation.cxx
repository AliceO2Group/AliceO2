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
#include "VecGeomParallelWorld.h"
#include "VecGeomG4PropagatingNavigator.h"

#include "TG4RootDetectorConstruction.h"
#include "TG4RootNavMgr.h"

#include "G4EventManager.hh"
#include "G4FieldManager.hh"
#include "G4PropagatorInField.hh"
#include "G4SteppingManager.hh"
#include "G4TrackingManager.hh"
#include "G4TransportationManager.hh"
#include "G4Threading.hh"

#include "TMCManager.h"
#include "TStopwatch.h"

#endif

namespace o2::simsetup
{

#ifdef O2_WITH_VECGEOM

bool isVecGeomNavigationAvailable() { return true; }

namespace
{
// The VecGeom side built once by installVecGeomNavigator; read-only afterwards and shared by the navigators of all threads.
struct SharedVecGeomSetup {
  VecGeomG4Map map;
  VecGeomParallelWorld parallelWorldStore;
  VecGeomParallelWorld const* parallelWorld = nullptr;
  bool leanLocate = false;
  bool built = false;
};
SharedVecGeomSetup gShared;

// Creates a VecGeom navigator and makes it the tracking navigator of the calling thread.
void registerNavigator(TG4RootDetectorConstruction& detConstruction)
{
  auto const& g4Params = o2::conf::G4Params::Instance();
  G4Navigator* navigator = nullptr;
  if (g4Params.vecgeomNavigator == o2::conf::EVecGeomNav::kPropagated) {
    LOG(info) << "VecGeom navigation: propagating navigator";
    navigator = new VecGeomG4PropagatingNavigator(gShared.map, g4Params.vecgeomZeroSafety);
  } else {
    LOG(info) << "VecGeom relocating navigator with the " << (gShared.leanLocate ? "lean" : "previous") << " locate";
    navigator = new VecGeomG4Navigator(gShared.map, g4Params.vecgeomPushDepth, g4Params.vecgeomZeroSafety,
                                       gShared.parallelWorld, gShared.leanLocate);
  }
  navigator->SetWorldVolume(detConstruction.GetTopPV());

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
  LOG(info) << "VecGeom navigator registered with the Geant4 transportation manager"
            << (G4Threading::IsWorkerThread() ? " of a worker thread" : "");
}
} // namespace

void installVecGeomNavigator()
{
  auto const& g4Params = o2::conf::G4Params::Instance();

  if (TMCManager::Instance() != nullptr) {
    LOG(fatal) << "G4.navmode=kVecGeom cannot be used with the multi-engine TMCManager: restoring a "
                  "geometry state across engines goes through the TGeo navigator";
  }
  const bool useParallelWorld = o2::GeometryManagerParam::Instance().useParallelWorld;
  if (useParallelWorld && g4Params.vecgeomNavigator != o2::conf::EVecGeomNav::kRelocating) {
    LOG(fatal) << "GeometryManagerParam.useParallelWorld needs G4.vecgeomNavigator=kRelocating: the propagating "
                  "navigator adopts the state VecGeom propagated, which knows no priority world";
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

  if (!g4Params.vecgeomFlattenAssemblies && g4Params.vecgeomNavigator == o2::conf::EVecGeomNav::kRelocating) {
    LOG(fatal) << "G4.vecgeomNavigator=kRelocating needs G4.vecgeomFlattenAssemblies=true: it enters a daughter by "
                  "locating inside it, which an assembly cannot answer";
  }

  TStopwatch timer;
  timer.Start();
  o2::base::GeometryManager::buildVecGeomGeometry(g4Params.vecgeomFlattenAssemblies, g4Params.vecgeomBooleanThreshold);
  timer.Stop();
  LOG(info) << "VecGeom geometry built in " << timer.RealTime() << " s";
  // The lean locate needs every mother with daughters stepped with the BVH navigator; kPropagated keeps the build's.
  const bool leanLocate = g4Params.vecgeomLeanLocate && g4Params.vecgeomNavigator == o2::conf::EVecGeomNav::kRelocating;
  o2::base::GeometryManager::setVecGeomBvhForSmallMothers(leanLocate);

  timer.Start();
  auto& map = gShared.map;
  map.build(*detConstruction, g4Params.vecgeomFlattenAssemblies);
  timer.Stop();
  LOG(info) << "VecGeom to Geant4 map built in " << timer.RealTime() << " s";

  // The prioritised placements of the TGeo parallel world, navigated with VecGeom.
  auto& parallelWorld = gShared.parallelWorldStore;
  VecGeomParallelWorld const* parallelWorldPtr = nullptr;
  if (useParallelWorld) {
    if (parallelWorld.build(*detConstruction, map)) {
      parallelWorldPtr = &parallelWorld;
    } else {
      LOG(warning) << "GeometryManagerParam.useParallelWorld is set but the TGeo parallel world is empty";
    }
  }

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

  gShared.leanLocate = leanLocate;
  gShared.parallelWorld = parallelWorldPtr;
  gShared.built = true;
  registerNavigator(*detConstruction);
}

void installVecGeomNavigatorOnWorker()
{
  if (!G4Threading::IsWorkerThread()) {
    LOG(fatal) << "installVecGeomNavigatorOnWorker must be called on a Geant4 worker thread";
  }
  if (!gShared.built) {
    LOG(fatal) << "The VecGeom geometry has not been built; installVecGeomNavigator must run on the master first";
  }
  auto* navMgr = TG4RootNavMgr::GetInstance(); // this thread's, cloned by geant4_vmc from the master's
  if (navMgr == nullptr || navMgr->GetDetConstruction() == nullptr) {
    LOG(fatal) << "G4.navmode=kVecGeom found no g4root navigation on this worker thread";
  }
  registerNavigator(*navMgr->GetDetConstruction());
}

#else

bool isVecGeomNavigationAvailable() { return false; }

void installVecGeomNavigator()
{
  LOG(fatal) << "G4.navmode=kVecGeom needs O2 built against TGeo2VecGeom and a VecGeom with the BVH navigator of "
                "the VNavigator family (BVHNavigatorV), which were not found at configure time";
}

void installVecGeomNavigatorOnWorker() { installVecGeomNavigator(); }

#endif

} // namespace o2::simsetup
