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

#include "VecGeomChecks.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TRandom3.h"

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <fairlogger/Logger.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

using V3 = vecgeom::Vector3D<double>;

namespace o2::simsetup
{

std::size_t checkVecGeomLocation(std::size_t samples)
{
  auto* world = vecgeom::GeoManager::Instance().GetWorld();
  auto const* box = dynamic_cast<TGeoBBox const*>(gGeoManager->GetTopVolume()->GetShape());
  if (world == nullptr || box == nullptr) {
    LOG(warning) << "Cannot cross-check the VecGeom location: no world";
    return 0;
  }
  TRandom3 rnd(12345);
  vecgeom::NavigationState state;
  std::map<std::string, std::size_t> byVolume;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < samples; ++i) {
    const double x = box->GetOrigin()[0] + box->GetDX() * (2. * rnd.Rndm() - 1.);
    const double y = box->GetOrigin()[1] + box->GetDY() * (2. * rnd.Rndm() - 1.);
    const double z = box->GetOrigin()[2] + box->GetDZ() * (2. * rnd.Rndm() - 1.);
    auto* node = gGeoManager->FindNode(x, y, z);
    const std::string tgeoName = (node != nullptr) ? node->GetVolume()->GetName() : "<outside>";
    state.Clear();
    vecgeom::GlobalLocator::LocateGlobalPoint(world, V3(x, y, z), state, true);
    auto const* top = state.Top();
    const std::string vgName = (top != nullptr) ? top->GetLogicalVolume()->GetName() : "<outside>";
    if (tgeoName != vgName) {
      ++bad;
      ++byVolume[tgeoName + " -> " + vgName];
    }
  }
  LOG(info) << "VecGeom location cross-check: " << bad << " of " << samples << " points land in a "
            << "different volume than TGeo puts them in";
  std::vector<std::pair<std::size_t, std::string>> worst;
  for (auto const& e : byVolume) {
    worst.emplace_back(e.second, e.first);
  }
  std::sort(worst.rbegin(), worst.rend());
  for (std::size_t i = 0; i < worst.size() && i < 15; ++i) {
    LOG(info) << "  " << worst[i].first << "  " << worst[i].second;
  }
  return bad;
}

std::size_t checkVecGeomVolume(const char* name, std::size_t perPlacement, std::size_t maxPlacements)
{
  auto* world = vecgeom::GeoManager::Instance().GetWorld();
  auto* vol = gGeoManager->GetVolume(name);
  if (world == nullptr || vol == nullptr) {
    LOG(warning) << "Cannot cross-check volume " << name << ": not in the geometry";
    return 0;
  }
  auto const* box = dynamic_cast<TGeoBBox const*>(vol->GetShape());
  if (box == nullptr) {
    LOG(warning) << "Cannot cross-check volume " << name << ": its shape has no bounding box";
    return 0;
  }
  LOG(info) << "VecGeom sizes: " << vecgeom::GeoManager::Instance().GetRegisteredVolumesCount()
            << " logical, " << vecgeom::GeoManager::Instance().GetPlacedVolumesCount() << " placed, "
            << vecgeom::VPlacedVolume::GetIdCount() << " ids handed out";
  // Divisions and other generated placements are where the two trees are most likely to differ
  // in shape rather than in position, so report the daughter counts before sampling anything.
  if (auto* lv = vecgeom::GeoManager::Instance().FindLogicalVolume(name)) {
    LOG(info) << "  " << name << ": TGeo " << vol->GetNdaughters() << " daughters, VecGeom "
              << lv->GetDaughters().size();
  }
  {
    TIter nextVol(gGeoManager->GetListOfVolumes());
    TGeoVolume* mother = nullptr;
    while ((mother = static_cast<TGeoVolume*>(nextVol())) != nullptr) {
      bool found = false;
      for (int i = 0; i < mother->GetNdaughters(); ++i) {
        if (mother->GetNode(i)->GetVolume() == vol) {
          found = true;
          break;
        }
      }
      if (!found) {
        continue;
      }
      auto* mlv = vecgeom::GeoManager::Instance().FindLogicalVolume(mother->GetName());
      LOG(info) << "  mother " << mother->GetName() << ": TGeo " << mother->GetNdaughters()
                << " daughters, VecGeom " << (mlv != nullptr ? (long)mlv->GetDaughters().size() : -1)
                << (mother->GetFinder() != nullptr ? "  (divided)" : "");
      break;
    }
  }
  TRandom3 rnd(4321);
  vecgeom::NavigationState state;
  std::map<std::string, std::size_t> byResult;
  std::size_t placements = 0, tested = 0, bad = 0, badTransform = 0;
  double worstTransform = 0.;
  TGeoIterator it(gGeoManager->GetTopVolume());
  TGeoNode* node = nullptr;
  while ((node = it.Next()) != nullptr && placements < maxPlacements) {
    if (node->GetVolume() != vol) {
      continue;
    }
    ++placements;
    TGeoHMatrix matrix = *it.GetCurrentMatrix();
    for (std::size_t k = 0; k < perPlacement; ++k) {
      double local[3] = {box->GetOrigin()[0] + box->GetDX() * (2. * rnd.Rndm() - 1.),
                         box->GetOrigin()[1] + box->GetDY() * (2. * rnd.Rndm() - 1.),
                         box->GetOrigin()[2] + box->GetDZ() * (2. * rnd.Rndm() - 1.)};
      if (!vol->GetShape()->Contains(local)) {
        continue;
      }
      double global[3];
      matrix.LocalToMaster(local, global);
      ++tested;
      auto* found = gGeoManager->FindNode(global[0], global[1], global[2]);
      const std::string tgeoName = (found != nullptr) ? found->GetVolume()->GetName() : "<outside>";
      state.Clear();
      vecgeom::GlobalLocator::LocateGlobalPoint(world, V3(global[0], global[1], global[2]), state, true);
      auto const* top = state.Top();
      const std::string vgName = (top != nullptr) ? top->GetLogicalVolume()->GetName() : "<outside>";
      if (tgeoName != vgName) {
        ++bad;
        ++byResult[tgeoName + " -> " + vgName];
        continue;
      }
      // The volume is right; check that the state also composes the right transform. That goes
      // through the navigation index table, which is built separately from the daughter lists
      // the locators use, so it can be wrong where containment looks perfect.
      vecgeom::Transformation3D trans;
      state.TopMatrix(trans);
      const auto vgLocal = trans.Transform(V3(global[0], global[1], global[2]));
      const double d = std::sqrt((vgLocal[0] - local[0]) * (vgLocal[0] - local[0]) +
                                 (vgLocal[1] - local[1]) * (vgLocal[1] - local[1]) +
                                 (vgLocal[2] - local[2]) * (vgLocal[2] - local[2]));
      worstTransform = std::max(worstTransform, d);
      if (d > 1.e-6) {
        ++badTransform;
      }
    }
  }
  LOG(info) << "VecGeom volume cross-check " << name << ": worst local-point deviation "
            << worstTransform << " cm, " << badTransform << " points above 1e-6 cm";
  LOG(info) << "VecGeom volume cross-check " << name << ": " << placements << " placements, " << tested
            << " points, " << bad << " located differently than TGeo";
  for (auto const& e : byResult) {
    LOG(info) << "  " << e.second << "  " << e.first;
  }
  return bad;
}

void checkVecGeomRays(std::size_t rays)
{
  auto* world = vecgeom::GeoManager::Instance().GetWorld();
  if (world == nullptr) {
    return;
  }
  TRandom3 rnd(97531);
  std::map<std::string, std::size_t> tgeoSeen, vgSeen;
  constexpr std::size_t kMaxSteps = 20000;

  for (std::size_t r = 0; r < rays; ++r) {
    const double cost = 2. * rnd.Rndm() - 1.;
    const double sint = std::sqrt(1. - cost * cost);
    const double phi = 2. * M_PI * rnd.Rndm();
    const double dir[3] = {sint * std::cos(phi), sint * std::sin(phi), cost};

    gGeoManager->InitTrack(0., 0., 0., dir[0], dir[1], dir[2]);
    for (std::size_t k = 0; k < kMaxSteps && !gGeoManager->IsOutside(); ++k) {
      ++tgeoSeen[gGeoManager->GetCurrentVolume()->GetName()];
      gGeoManager->FindNextBoundaryAndStep();
    }

    vecgeom::NavigationState cur, next;
    V3 pos(0., 0., 0.);
    const V3 vdir(dir[0], dir[1], dir[2]);
    vecgeom::GlobalLocator::LocateGlobalPoint(world, pos, cur, true);
    for (std::size_t k = 0; k < kMaxSteps && cur.Top() != nullptr; ++k) {
      ++vgSeen[cur.Top()->GetLogicalVolume()->GetName()];
      double safety = 0.;
      auto const* nav = cur.Top()->GetLogicalVolume()->GetNavigator();
      const double step = nav->ComputeStepAndSafetyAndPropagatedState(pos, vdir, vecgeom::kInfLength, cur, next,
                                                                      false, safety);
      if (!(step < vecgeom::kInfLength)) {
        break;
      }
      pos = pos + step * vdir;
      cur = next;
    }
  }

  // A volume one engine never enters is the sharpest signal: a hit can only be made in a
  // volume a track actually reaches, so these are the ones that lose a detector its hits.
  std::vector<std::string> neverVG, neverTGeo;
  std::vector<std::pair<long, std::string>> diff;
  for (auto const& e : tgeoSeen) {
    const std::size_t vg = vgSeen.count(e.first) ? vgSeen[e.first] : 0;
    if (vg == 0) {
      neverVG.push_back(e.first);
    }
    const long d = static_cast<long>(e.second) - static_cast<long>(vg);
    if (d != 0) {
      diff.emplace_back(std::labs(d), e.first);
    }
  }
  for (auto const& e : vgSeen) {
    if (tgeoSeen.count(e.first) == 0) {
      neverTGeo.push_back(e.first);
    }
  }
  std::sort(diff.rbegin(), diff.rend());
  LOG(info) << "VecGeom ray cross-check over " << rays << " rays: " << tgeoSeen.size() << " volumes seen by TGeo, "
            << vgSeen.size() << " by VecGeom, " << diff.size() << " entered a different number of times";
  LOG(info) << "  never entered by VecGeom (" << neverVG.size() << "):";
  for (std::size_t i = 0; i < neverVG.size() && i < 60; ++i) {
    LOG(info) << "    " << neverVG[i] << " (TGeo " << tgeoSeen[neverVG[i]] << ")";
  }
  LOG(info) << "  never entered by TGeo (" << neverTGeo.size() << "):";
  for (std::size_t i = 0; i < neverTGeo.size() && i < 30; ++i) {
    LOG(info) << "    " << neverTGeo[i] << " (VecGeom " << vgSeen[neverTGeo[i]] << ")";
  }
}

} // namespace o2::simsetup
