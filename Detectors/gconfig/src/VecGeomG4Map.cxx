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

#include "VecGeomG4Map.h"

#include "TG4RootDetectorConstruction.h"

#include "G4PhysicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TString.h"

#include "TGeo2VecGeom/RootGeoManager.h"
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/LogicalVolume.h>

#include <fairlogger/Logger.h>

#include <algorithm>
#include <functional>
#include <unordered_set>

namespace o2::simsetup
{

namespace
{
/// Reproduces tgeo2vecgeom's assembly flattening for the daughters of one TGeo volume, so that
/// each VecGeom daughter can be paired with the chain of TGeo nodes it stands for. The
/// converter walks a volume's daughters in order and, when flattening, descends into an
/// assembly instead of placing it, appending the leaves it reaches; the placed daughters of
/// the VecGeom logical volume come out in exactly that order.
void flattenDaughter(TGeoNode* node, bool flatten, std::vector<TGeoNode*>& chain,
                     std::vector<std::vector<TGeoNode*>>& out)
{
  chain.push_back(node);
  auto* assembly = dynamic_cast<TGeoVolumeAssembly*>(node->GetVolume());
  if (flatten && assembly != nullptr) {
    for (int i = 0; i < assembly->GetNdaughters(); ++i) {
      flattenDaughter(assembly->GetNode(i), flatten, chain, out);
    }
  } else {
    out.push_back(chain);
  }
  chain.pop_back();
}

std::vector<std::vector<TGeoNode*>> flattenedDaughters(TGeoVolume const* volume, bool flatten)
{
  std::vector<std::vector<TGeoNode*>> out;
  std::vector<TGeoNode*> chain;
  for (int i = 0; i < volume->GetNdaughters(); ++i) {
    flattenDaughter(volume->GetNode(i), flatten, chain, out);
  }
  return out;
}
} // namespace

void VecGeomG4Map::registerPair(vecgeom::VPlacedVolume const* parent, vecgeom::VPlacedVolume const* pv,
                                std::vector<TGeoNode*> const& nodes, TG4RootDetectorConstruction const& dc)
{
  const auto vgId = static_cast<std::size_t>(pv->id());
  if (vgId >= mChainBegin.size()) {
    mChainBegin.resize(vgId + 1, 0);
    mChainSize.resize(vgId + 1, 0);
  }
  if (mChainSize[vgId] != 0) { // a logical volume placed more than once shares its daughters
    return;
  }
  mChainBegin[vgId] = static_cast<unsigned>(mChain.size());
  bool first = true;
  mChainSize[vgId] = static_cast<unsigned>(nodes.size());
  for (auto* node : nodes) {
    auto* g4pv = dc.GetG4VPhysicalVolume(node);
    if (g4pv == nullptr) {
      LOG(fatal) << "TGeo node " << node->GetName() << " has no Geant4 counterpart; the two conversions "
                 << "of the geometry do not agree";
    }
    mChain.push_back(g4pv);
    if (first && parent == nullptr) {
      mWorld = pv;
    }
    first = false;
    if (node != nodes.back()) {
      continue; // an intermediate assembly level, dissolved on the VecGeom side
    }
    const auto g4Id = static_cast<std::size_t>(g4pv->GetInstanceID());
    if (g4Id >= mG4ToVG.size()) {
      mG4ToVG.resize(g4Id + 1, nullptr);
    }
    if (mG4ToVG[g4Id] == nullptr) {
      mG4ToVG[g4Id] = pv;
    } else if (mG4ToVG[g4Id] == ambiguous()) {
      mAmbiguous[static_cast<int>(g4Id)].push_back(pv);
    } else if (mG4ToVG[g4Id] != pv) {
      // Reached through more than one flattened chain: the content of an assembly placed in
      // more than one place. They are told apart by the chain itself, not by this volume.
      auto& list = mAmbiguous[static_cast<int>(g4Id)];
      list.push_back(mG4ToVG[g4Id]);
      list.push_back(pv);
      mG4ToVG[g4Id] = ambiguous();
    }
  }
  ++mPairs;
}

void VecGeomG4Map::build(TG4RootDetectorConstruction const& dc, bool flattenAssemblies)
{
  auto& rootGeoMgr = tgeo2vecgeom::RootGeoManager::Instance();
  auto& vgMgr = vecgeom::GeoManager::Instance();
  // Ids are handed out by a global counter, so the largest one can exceed the number of
  // volumes the manager holds; sizing by the count alone reads past the end.
  const auto reserve = std::max<std::size_t>(vgMgr.GetPlacedVolumesCount(), vecgeom::VPlacedVolume::GetIdCount()) + 1;
  mChainBegin.assign(reserve, 0);
  mChainSize.assign(reserve, 0);
  mChain.reserve(reserve);
  mG4ToVG.assign(G4PhysicalVolumeStore::GetInstance()->size() + 1, nullptr);

  auto* topNode = gGeoManager->GetTopNode();
  auto const* topPV = rootGeoMgr.Lookup(topNode);
  if (topPV == nullptr) {
    LOG(fatal) << "The VecGeom geometry has no counterpart for the TGeo top node";
  }
  registerPair(nullptr, topPV, {topNode}, dc);

  // A TGeoVolume placed twice shares one set of daughter nodes, and so do both conversions of
  // it, so each volume's daughters are paired up exactly once.
  std::unordered_set<TGeoVolume const*> seen;
  std::function<void(vecgeom::VPlacedVolume const*, TGeoNode*)> walk =
    [&](vecgeom::VPlacedVolume const* pv, TGeoNode* node) {
      auto* volume = node->GetVolume();
      if (!seen.insert(volume).second) {
        return;
      }
      const auto chains = flattenedDaughters(volume, flattenAssemblies);
      auto const& vgDaughters = pv->GetLogicalVolume()->GetDaughters();
      if (vgDaughters.size() != chains.size()) {
        LOG(fatal) << "Volume " << volume->GetName() << " has " << chains.size()
                   << " daughters after flattening but its VecGeom counterpart has " << vgDaughters.size()
                   << "; the flattening reproduced here does not match the converter's";
      }
      for (std::size_t i = 0; i < chains.size(); ++i) {
        auto const* daughter = vgDaughters[i];
        // The pairing is by position, so check it against what the converter recorded. The
        // converter names a node it synthesised while flattening after the original one, so
        // the recorded node must be either that node itself or a flattened copy of it.
        auto const* recorded = rootGeoMgr.tgeonode(daughter);
        auto const* expected = chains[i].back();
        const bool ok = recorded != nullptr &&
                        (recorded == expected ||
                         (recorded->GetVolume() == expected->GetVolume() &&
                          TString(recorded->GetName()).BeginsWith(TString(expected->GetName()) + "_assemblyinternalcount_")));
        if (!ok) {
          ++mMispaired;
          if (mMispaired <= 5) {
            LOG(warning) << "VecGeom daughter " << i << " of " << volume->GetName() << " is "
                         << (recorded != nullptr ? recorded->GetName() : "unknown") << ", expected "
                         << expected->GetName();
          }
        }
        registerPair(pv, daughter, chains[i], dc);
        walk(daughter, chains[i].back());
      }
    };
  walk(topPV, topNode);

  if (mMispaired != 0) {
    LOG(error) << mMispaired << " VecGeom placements were paired with the wrong TGeo node; the "
               << "order the converter places flattened daughters in is not the order assumed here";
  }
  LOG(info) << "VecGeom navigation: paired " << mPairs << " placements with " << mChain.size()
            << " Geant4 volumes (flattenAssemblies=" << flattenAssemblies << "), " << ambiguousCount()
            << " of them shared by more than one placement";
}

} // namespace o2::simsetup
