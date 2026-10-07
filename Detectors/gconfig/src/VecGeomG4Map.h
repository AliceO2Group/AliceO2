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

#include <VecGeom/volumes/PlacedVolume.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class G4VPhysicalVolume;
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

} // namespace o2::simsetup

#endif
