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

#ifndef ALICEO2_CADSUPPORT_O2FLATCSG_
#define ALICEO2_CADSUPPORT_O2FLATCSG_

#include "TGeoBBox.h"

#include <vector>

namespace o2
{
namespace cad
{

/// One signed implicit halfspace, the region `sign * f(x) <= 0`: kQuadric stores `x^T A x + 2 b^T x + c` as
/// (a00, a01, a02, a11, a12, a22, b0, b1, b2, c); kTorus stores (px, py, pz, dx, dy, dz, R, r) in the first eight.
struct FlatCSGHalfspace {
  enum Kind : int { kQuadric = 0,
                    kTorus = 1 };
  int kind = kQuadric;
  double sign = 1.;
  double c[11] = {};
};

/// One DNF cell: `[first, first + count)` of the halfspace array, intersected; `volume` is its own volume.
struct FlatCSGCell {
  int first = 0;
  int count = 0;
  double volume = 0.;
};

/// One box of the sub-cell subdivision; `nActive == 0` means it is wholly inside its cell.
/// An active list describes its cell only inside its box, so every ray query clips to the box first.
struct FlatCSGBox {
  double min[3] = {};
  double max[3] = {};
  int cell = -1;
  int firstActive = 0;
  int nActive = 0;
};

/// A solid stored as a union of intersection cells over signed implicit halfspaces, the flat DNF of a decomposed part.
/// Every accelerated query has a bit-identical `_Loop` twin over all cells and halfspaces.
class O2FlatCSG : public TGeoBBox
{
 public:
  O2FlatCSG();
  explicit O2FlatCSG(const char* name);
  ~O2FlatCSG() override;

  // The shape owns a raw `bvh::v2::Bvh` behind `fBVH`, so a compiler-written copy would hand two
  // shapes the same BVH and then free it twice; same treatment as O2BVHAssembly.
  O2FlatCSG(const O2FlatCSG&) = delete;
  O2FlatCSG& operator=(const O2FlatCSG&) = delete;

  // ---- building -------------------------------------------------------------------------
  /// Append a quadric halfspace; returns its index. `sign` is +1 or -1, inside is `sign*Q <= 0`.
  int AddQuadric(double sign, const double coeff[10]);
  /// Append a torus halfspace, inside `sign * (sqrt((rho - major)^2 + z^2) - minor) <= 0` about unit \a axis; returns its index.
  int AddTorus(double sign, const double* centre, const double* axis, double major, double minor);
  /// Append a cell over `[first, first + count)` of the halfspace array; returns its index.
  int AddCell(int first, int count, double volume);

  int GetNhalfspaces() const { return static_cast<int>(fHalfspaces.size()); }
  int GetNcells() const { return static_cast<int>(fCells.size()); }
  const FlatCSGHalfspace& GetHalfspace(int index) const { return fHalfspaces[index]; }
  const FlatCSGCell& GetCell(int index) const { return fCells[index]; }

  /// The AABB of cell \a cell. The halfspaces alone do not bound a cell -- an intersection of
  /// halfspaces can be unbounded -- so the converter supplies the box the decomposition measured.
  void SetCellBBox(int cell, const double* lo, const double* hi);
  /// The AABB `SetCellBBox` recorded for cell \a cell, for the sidecar writer. Reads back zeros
  /// for a cell whose box was never set.
  void GetCellBBox(int cell, double* lo, double* hi) const;

  /// Build the sub-cell boxes and their BVH. Call once, after the last AddCell.
  void CloseShape();
  bool IsClosed() const { return fClosed; }

  /// Bytes held by the BVH nodes and the primitive-index permutation.
  size_t GetBVHMemory() const;

  int GetNboxes() const { return static_cast<int>(fBoxes.size()); }
  const FlatCSGBox& GetBox(int index) const { return fBoxes[index]; }
  /// For the tests: the box structure is the thing being proved sound, so it has to be readable.
  int GetActive(int index) const { return fActive[index]; }
  /// True when every halfspace of cell `index` contains `point`.
  bool CellContains(int index, const double* point) const;

  /// Subdivision depth cap. See `fSplitDepth` for where the default comes from.
  void SetSplitDepth(int depth) { fSplitDepth = depth; }
  /// Stop splitting a box narrower than this fraction of the part's bounding-box diagonal.
  /// See `fMinBoxFraction` for where the default comes from.
  void SetMinBoxFraction(double fraction) { fMinBoxFraction = fraction; }

  /// `sign * f(point)`; the halfspace contains the point when this is `<= 0`.
  static double EvalHalfspace(const FlatCSGHalfspace& halfspace, const double* point);

  /// A rigorous enclosure `[rangeLo, rangeHi]` of `sign * f` over the box `[lo, hi]`, padded outward.
  /// Requires `lo[i] <= hi[i]` and finite bounds, which CloseShape enforces; it does not check them.
  static void HalfspaceRange(const FlatCSGHalfspace& halfspace, const double* lo, const double* hi,
                             double& rangeLo, double& rangeHi);

  /// Real roots of `sign * f(origin + t*dir) = 0`, unsorted, at most four; returns the count.
  static int HalfspaceRoots(const FlatCSGHalfspace& halfspace, const double* origin,
                            const double* dir, double* roots);

  /// The occupancy of cell \a cell along the ray within `[tlo, thi]`, as `[enter, exit]` pairs in \a out; a null \a active uses every halfspace.
  /// Returns the pair count, or a negative value when \a maxOut is too small.
  int CellIntervals(int cell, const int* active, int nActive, const double* origin,
                    const double* dir, double tlo, double thi, double* out, int maxOut) const;

  // ---- the TGeoShape contract; the accelerated queries use their `_Loop` twin until CloseShape succeeds ----
  Bool_t Contains(const Double_t* point) const override;

  Double_t DistFromOutside(const Double_t* point, const Double_t* dir, Int_t iact = 1,
                           Double_t step = TGeoShape::Big(), Double_t* safe = nullptr) const override;
  Double_t DistFromInside(const Double_t* point, const Double_t* dir, Int_t iact = 1,
                          Double_t step = TGeoShape::Big(), Double_t* safe = nullptr) const override;

  /// A lower bound on the distance to the boundary from the box structure: outside the nearest box, inside the faces of a solid box, else 0.
  Double_t Safety(const Double_t* point, Bool_t in = kTRUE) const override;

  /// Per-thread count of DistFromInside queries whose pruned traversal had to be redone unpruned.
  static void ResetUnprunedRetryCounter();
  static long long GetUnprunedRetryCount();

  /// The union of the retained sub-cell boxes, tighter than the union of the cell AABBs.
  void ComputeBBox() override;

  /// The sum of the cells' own volumes. The cells of a decomposition are disjoint by construction
  /// (`decompose`'s volume guard checks it), so there is no inclusion-exclusion to do.
  Double_t Capacity() const override;

  /// The normal of the halfspace nearest to equality at `point`, oriented along `dir`.
  void ComputeNormal(const Double_t* point, const Double_t* dir, Double_t* norm) const override;
  /// Points on the solid's own boundary, for the overlap checkers; kFALSE if fewer than \a npoints were found.
  Bool_t GetPointsOnSegments(Int_t npoints, Double_t* array) const override;

  // ---- the reference twins ---------------------------------------------------------------
  Bool_t Contains_Loop(const Double_t* point) const;

  Double_t DistFromOutside_Loop(const Double_t* point, const Double_t* dir,
                                Double_t step = TGeoShape::Big()) const;
  Double_t DistFromInside_Loop(const Double_t* point, const Double_t* dir,
                               Double_t step = TGeoShape::Big()) const;
  /// `Safety`'s twin over all boxes: it must equal `Safety` and be a sound bound.
  Double_t Safety_Loop(const Double_t* point, Bool_t in = kTRUE) const;

 protected:
  /// Grow the per-cell bounding-box storage to the cell count.
  void EnsureCellBBoxStorage();

  /// The accelerated DistFromOutside/DistFromInside bodies; each clips the ray to a box before using its active list.
  Double_t DistFromOutsideBVH(const Double_t* point, const Double_t* dir, Double_t step) const;
  Double_t DistFromInsideBVH(const Double_t* point, const Double_t* dir, Double_t step) const;

  /// What the running bound prunes against: nothing, the nearest entry so far (DistFromOutside) or
  /// the far end of the interval holding t = 0 so far (DistFromInside).
  enum class RayBound { kNone,
                        kEntry,
                        kExit };

  /// Each box's own occupancy pieces along the ray within `[0, step]`: `[enter, exit]` in \a pairs and its cell in \a cells, unmerged.
  /// False when a `CellIntervals` call overflowed. \a smallestPruned reports the nearest entry the
  /// exit bound skipped, Big if it skipped nothing or if the bound is not `kExit`.
  bool GatherRayPieces(const Double_t* point, const Double_t* dir, Double_t step,
                       std::vector<double>& pairs, std::vector<int>& cells, RayBound bound,
                       double& smallestPruned) const;

  /// Recursively split `[lo, hi]` for `cell`, dropping the halfspaces the range bound decides and the boxes it proves outside.
  /// A split of a far-from-cubic box draws on `cubifyBudget`, any other on `depth`.
  void SplitBox(int cell, const double* lo, const double* hi, const std::vector<int>& active,
                int depth, double minSize, int cubifyBudget);

  std::vector<FlatCSGHalfspace> fHalfspaces; ///< the flat halfspace array
  std::vector<FlatCSGCell> fCells;           ///< the DNF's cells, indexing into it

  /// The sub-cell boxes, rebuilt by `CloseShape`; not streamed.
  std::vector<FlatCSGBox> fBoxes; //!
  /// The boxes' active-halfspace lists, concatenated. Derived alongside `fBoxes`; not streamed
  /// for the same reason.
  std::vector<int> fActive;    //!
  std::vector<double> fCellLo; ///< each cell's AABB low corner, 3 doubles per cell
  std::vector<double> fCellHi; ///< each cell's AABB high corner, 3 doubles per cell
  /// Whether `SetCellBBox` was ever called for a given cell; `CloseShape` refuses to build a
  /// solid missing one rather than silently drop that cell -- see `CloseShape`'s implementation.
  std::vector<bool> fCellBBoxSet;
  /// Set by a successful `CloseShape`; not streamed. The `#pragma read` rule closes every shape ROOT reads back.
  bool fClosed = false; //!
  /// Subdivision depth cap, and the minimum box size as a fraction of the part's bounding-box
  /// diagonal, chosen for query cost on the shipped parts.
  int fSplitDepth = 4;
  double fMinBoxFraction = 0.05;

  /// The BVH over `fBoxes`, rebuilt by `CloseShape`; not streamed.
  void* fBVH = nullptr; //! bvh::v2::Bvh over the sub-cell boxes

  // Scratch buffers are thread_local statics in the .cxx, never members: shapes are shared by all navigator threads.

  ClassDefOverride(O2FlatCSG, 1) // flat-DNF halfspace shape class
};

} // namespace cad
} // namespace o2

#endif
