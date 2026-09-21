// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Sandro Wenzel <sandro.wenzel@cern.ch>
/// \since 2026-08

#ifndef ALICEO2_CADSUPPORT_O2BVHASSEMBLY_
#define ALICEO2_CADSUPPORT_O2BVHASSEMBLY_

#include "TGeoShapeAssembly.h"

class TGeoVolumeAssembly;

namespace o2
{
namespace cad
{

/// A BVH-accelerated drop-in for ROOT's `TGeoShapeAssembly`: a BVH over the daughter boxes answers Contains, DistFromOutside and Safety.
/// Install it with MakeBVHAssembly(volume) after `TGeoManager::CloseGeometry()`, or construct it on the volume and call `SetShape`.
/// Each query has a `_Loop` twin over all daughters in index order; the lowest-indexed daughter wins ties, as in ROOT.
/// Unlike ROOT, DistFromOutside also answers from outside the bounding box of a voxelized assembly.
class O2BVHAssembly : public TGeoShapeAssembly
{
 public:
  O2BVHAssembly();
  /// Build over the daughters \a volume has *now*; later daughters trigger a lazy rebuild.
  explicit O2BVHAssembly(TGeoVolumeAssembly* volume);
  ~O2BVHAssembly() override;

  O2BVHAssembly(const O2BVHAssembly&) = delete;
  O2BVHAssembly& operator=(const O2BVHAssembly&) = delete;

  /// (Re)build the acceleration structure from the volume's current daughter list.
  void BuildBVH();
  /// Number of daughter placements the current BVH covers, -1 if it was never built.
  int GetNbuilt() const { return fNbuilt; }
  /// Bytes held by the BVH nodes and the primitive-index permutation.
  size_t GetBVHMemory() const;

  // ---- the accelerated part of the TGeoShapeAssembly contract ----------------------------
  Bool_t Contains(const Double_t* point) const override;
  Double_t DistFromOutside(const Double_t* point, const Double_t* dir, Int_t iact = 1,
                           Double_t step = TGeoShape::Big(), Double_t* safe = nullptr) const override;
  Double_t Safety(const Double_t* point, Bool_t in = kTRUE) const override;

  // ---- the reference twins: same answer, all daughters, index order ----------------------
  Bool_t Contains_Loop(const Double_t* point) const;
  Double_t DistFromOutside_Loop(const Double_t* point, const Double_t* dir,
                                Double_t step = TGeoShape::Big()) const;
  Double_t Safety_Loop(const Double_t* point, Bool_t in = kTRUE) const;

  /// Replace \a volume's shape by an O2BVHAssembly and return it.
  static O2BVHAssembly* MakeBVHAssembly(TGeoVolumeAssembly* volume);

 private:
  /// Rebuild if the daughter count changed since the last build; not thread-safe while the geometry is being assembled.
  void EnsureBuilt() const;

  void* fBVH = nullptr; //! bvh::v2::Bvh over the daughter placement boxes
  int fNbuilt = -1;     //! daughter count the BVH was built for; -1 = never built
  int fTreeDepth = 0;   //! node levels of the BVH, which size the traversal stacks

  ClassDefOverride(O2BVHAssembly, 1) // BVH-accelerated assembly shape
};

} // namespace cad
} // namespace o2

#endif
