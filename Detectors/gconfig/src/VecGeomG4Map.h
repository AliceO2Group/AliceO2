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

#ifndef O2_SIMSETUP_VECGEOMG4MAP_H_
#define O2_SIMSETUP_VECGEOMG4MAP_H_

#include "G4VPhysicalVolume.hh"

#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class TG4RootDetectorConstruction;
class TGeoNode;

namespace o2::simsetup
{

/// The correspondence between the VecGeom tree tgeo2vecgeom built and the Geant4 tree g4root
/// built. Both are conversions of the same TGeoNode hierarchy, but not necessarily level for
/// level: with assembly flattening one VecGeom placement stands for a whole assembly subtree,
/// so a placement maps to a *chain* of Geant4 physical volumes rather than to one. Pushing the
/// whole chain is what keeps the Geant4 touchable at the depth the TGeo navigator produces,
/// and hence keeps every volume id, copy number and CurrentVolOffName offset unchanged.
///
/// Without flattening every chain has length one. Lookups are vector indexing: VecGeom
/// placed-volume ids and Geant4 physical-volume instance ids are both dense.
class VecGeomG4Map
{
 public:
  /// Walks the TGeo hierarchy and pairs up the two conversions of it. Both must exist.
  /// \param flattenAssemblies must match what the converter was told, since it decides how
  /// many TGeo nodes stand behind one VecGeom placement.
  void build(TG4RootDetectorConstruction const& dc, bool flattenAssemblies);

  /// The Geant4 volumes a VecGeom placement stands for, outermost first. A size of zero means
  /// the placement was never paired up, which is a bug rather than a legal state.
  G4VPhysicalVolume* const* chain(int vgId, unsigned& size) const
  {
    if (static_cast<std::size_t>(vgId) >= mChainSize.size()) {
      size = 0;
      return nullptr;
    }
    size = mChainSize[vgId];
    return mChain.data() + mChainBegin[vgId];
  }

  /// The VecGeom placement whose chain ends at the Geant4 volume \a g4Id. Null means that
  /// volume is only ever an intermediate assembly level, which flattening dissolved on the
  /// VecGeom side, so the walk extends past it. A plain vector index; the few Geant4 volumes
  /// reachable through more than one flattened chain answer ambiguous() and go to candidates().
  vecgeom::VPlacedVolume const* toVecGeom(int g4Id) const { return mG4ToVG[g4Id]; }

  /// Marks a Geant4 volume that several VecGeom placements reach.
  static vecgeom::VPlacedVolume const* ambiguous()
  {
    return reinterpret_cast<vecgeom::VPlacedVolume const*>(std::uintptr_t{1});
  }

  /// The placements that share an ambiguous Geant4 volume, to be told apart by their chains.
  std::vector<vecgeom::VPlacedVolume const*> const& candidates(int g4Id) const
  {
    static const std::vector<vecgeom::VPlacedVolume const*> empty;
    const auto it = mAmbiguous.find(g4Id);
    return it == mAmbiguous.end() ? empty : it->second;
  }

  /// Whether a placement's chain is exactly the given run of Geant4 volumes.
  bool chainMatches(vecgeom::VPlacedVolume const* pv, G4VPhysicalVolume* const* first, std::size_t n) const
  {
    unsigned size = 0;
    auto* const* own = chain(pv->id(), size);
    if (size != n) {
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (own[i] != first[i]) {
        return false;
      }
    }
    return true;
  }

  /// Builds \a state from a Geant4 path, extending past the levels assembly flattening dissolved.
  /// \a volumeAt(level) gives the Geant4 volume at a level, 0 being the world, \a depth the deepest
  /// level. False if a level matches no VecGeom placement; \a state is then empty.
  template <typename VolumeAt>
  bool stateFromPath(VolumeAt&& volumeAt, std::size_t depth, vecgeom::NavigationState& state) const;

  /// How many Geant4 levels a VecGeom placement accounts for.
  unsigned chainSize(int vgId) const
  {
    return static_cast<std::size_t>(vgId) < mChainSize.size() ? mChainSize[vgId] : 0;
  }

  /// The outermost placement, standing for the Geant4 world volume.
  vecgeom::VPlacedVolume const* world() const { return mWorld; }

  /// How many Geant4 volumes share more than one VecGeom placement.
  std::size_t ambiguousCount() const { return mAmbiguous.size(); }

  std::size_t size() const { return mPairs; }

 private:
  void registerPair(vecgeom::VPlacedVolume const* parent, vecgeom::VPlacedVolume const* pv,
                    std::vector<TGeoNode*> const& nodes, TG4RootDetectorConstruction const& dc);

  std::vector<unsigned> mChainBegin;
  std::vector<unsigned> mChainSize;
  std::vector<G4VPhysicalVolume*> mChain;
  std::vector<vecgeom::VPlacedVolume const*> mG4ToVG;
  std::unordered_map<int, std::vector<vecgeom::VPlacedVolume const*>> mAmbiguous;
  vecgeom::VPlacedVolume const* mWorld = nullptr;
  std::size_t mPairs = 0;
  std::size_t mMispaired = 0;
};

template <typename VolumeAt>
bool VecGeomG4Map::stateFromPath(VolumeAt&& volumeAt, std::size_t depth, vecgeom::NavigationState& state) const
{
  // A run of levels that ends in a Geant4 volume several placements share is settled by comparing
  // the run against the placements' recorded chains, which is exact.
  constexpr std::size_t kMaxChain = 16; // longest run of Geant4 levels one flattened placement stands for
  state.Clear();
  if (mWorld == nullptr || volumeAt(0) == nullptr) {
    return false;
  }
  state.Push(mWorld);
  std::size_t l = chainSize(mWorld->id());
  while (l <= depth) {
    G4VPhysicalVolume* run[kMaxChain];
    vecgeom::VPlacedVolume const* found = nullptr;
    std::size_t n = 0;
    for (std::size_t e = l; e <= depth && n < kMaxChain; ++e, ++n) {
      run[n] = volumeAt(e);
      if (run[n] == nullptr) {
        state.Clear();
        return false;
      }
      auto const* candidate = toVecGeom(run[n]->GetInstanceID());
      if (candidate == nullptr) {
        continue; // an assembly level; the chain reaches further down
      }
      if (candidate == ambiguous()) {
        for (auto const* c : candidates(run[n]->GetInstanceID())) {
          if (chainMatches(c, run, n + 1)) {
            found = c;
            break;
          }
        }
      } else if (chainMatches(candidate, run, n + 1)) {
        found = candidate;
      }
      if (found != nullptr) {
        break;
      }
    }
    if (found == nullptr) {
      state.Clear();
      return false;
    }
    state.Push(found);
    l += chainSize(found->id());
  }
  return true;
}

} // namespace o2::simsetup

#endif
