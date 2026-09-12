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

#define BOOST_TEST_MODULE Test O2BVHAssembly class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "CADSupport/O2BVHAssembly.h"

#include "TFile.h"
#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMatrix.h"
#include "TGeoMedium.h"
#include "TGeoNode.h"
#include "TGeoShapeAssembly.h"
#include "TGeoVolume.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
using o2::cad::O2BVHAssembly;

/// A small deterministic generator, so a failing case can be reproduced from its seed alone.
class Rng
{
 public:
  explicit Rng(unsigned long long seed) : mState(seed) {}
  double uniform(double low, double high)
  {
    mState = mState * 6364136223846793005ULL + 1442695040888963407ULL;
    const double unit = static_cast<double>((mState >> 11) & ((1ULL << 53) - 1)) / static_cast<double>(1ULL << 53);
    return low + unit * (high - low);
  }
  void direction(double* dir)
  {
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        dir[index] = uniform(-1., 1.);
      }
      norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }
  }

 private:
  unsigned long long mState;
};

TGeoMedium* vacuum()
{
  auto* material = new TGeoMaterial("Vacuum", 0., 0., 0.);
  return new TGeoMedium("Vacuum", 1, material);
}

/// A fresh geometry holding one assembly of \a count^3 unit boxes on a \a pitch grid, inside a
/// world large enough that every query point can be placed by hand.
struct Grid {
  TGeoManager* manager = nullptr;
  TGeoVolume* world = nullptr;
  TGeoVolumeAssembly* assembly = nullptr;
  int count = 0;
  double pitch = 0.;
  double halfBox = 0.;
};

Grid makeGrid(const char* name, int count, double pitch, double halfBox)
{
  Grid grid;
  grid.manager = new TGeoManager(name, name);
  grid.count = count;
  grid.pitch = pitch;
  grid.halfBox = halfBox;
  auto* medium = vacuum();
  const double extent = 4. * count * pitch;
  grid.world = grid.manager->MakeBox("WORLD", medium, extent, extent, extent);
  grid.assembly = new TGeoVolumeAssembly("GRID");
  int copy = 0;
  for (int ix = 0; ix < count; ++ix) {
    for (int iy = 0; iy < count; ++iy) {
      for (int iz = 0; iz < count; ++iz) {
        auto* box = grid.manager->MakeBox(Form("cell_%d", copy), medium, halfBox, halfBox, halfBox);
        grid.assembly->AddNode(box, copy,
                               new TGeoTranslation(pitch * (ix - 0.5 * (count - 1)),
                                                   pitch * (iy - 0.5 * (count - 1)),
                                                   pitch * (iz - 0.5 * (count - 1))));
        ++copy;
      }
    }
  }
  grid.world->AddNode(grid.assembly, 1, new TGeoTranslation(0., 0., 0.));
  grid.manager->SetTopVolume(grid.world);
  return grid;
}

/// The extent a Grid's daughters occupy, half-width per axis.
double gridReach(const Grid& grid)
{
  return 0.5 * grid.pitch * (grid.count - 1) + grid.halfBox;
}

/// Clear the per-thread daughter indices through which the assembly queries report their daughter.
void clearNodeIndices(TGeoVolumeAssembly* volume)
{
  volume->SetCurrentNodeIndex(-1);
  volume->SetNextNodeIndex(-1);
}
} // namespace

// ---------------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(BuildsOnePrimitivePerDaughter)
{
  Grid grid = makeGrid("build_grid", 5, 3., 1.);
  auto* shape = new O2BVHAssembly(grid.assembly);
  BOOST_CHECK_EQUAL(shape->GetNbuilt(), 125);
  BOOST_CHECK_EQUAL(shape->GetNbuilt(), grid.assembly->GetNdaughters());
  BOOST_CHECK_GT(shape->GetBVHMemory(), 0u);
}

BOOST_AUTO_TEST_CASE(BoundingBoxMatchesRoot)
{
  Grid grid = makeGrid("bbox_grid", 4, 3., 1.);
  auto* rootShape = static_cast<TGeoShapeAssembly*>(grid.assembly->GetShape());
  rootShape->ComputeBBox();
  auto* shape = new O2BVHAssembly(grid.assembly);
  const auto* ours = static_cast<const TGeoBBox*>(shape);
  const auto* theirs = static_cast<const TGeoBBox*>(rootShape);
  BOOST_CHECK_EQUAL(ours->GetDX(), theirs->GetDX());
  BOOST_CHECK_EQUAL(ours->GetDY(), theirs->GetDY());
  BOOST_CHECK_EQUAL(ours->GetDZ(), theirs->GetDZ());
  for (int axis = 0; axis < 3; ++axis) {
    BOOST_CHECK_EQUAL(ours->GetOrigin()[axis], theirs->GetOrigin()[axis]);
  }
}

BOOST_AUTO_TEST_CASE(EmptyAssemblyAnswersNothing)
{
  auto* manager = new TGeoManager("empty_asm", "empty_asm");
  auto* medium = vacuum();
  auto* world = manager->MakeBox("WORLD", medium, 10., 10., 10.);
  auto* assembly = new TGeoVolumeAssembly("EMPTY");
  world->AddNode(assembly, 1, new TGeoTranslation(0., 0., 0.));
  manager->SetTopVolume(world);
  auto* shape = new O2BVHAssembly(assembly);
  BOOST_CHECK_EQUAL(shape->GetNbuilt(), 0);
  const double point[3] = {0., 0., 0.};
  const double direction[3] = {1., 0., 0.};
  BOOST_CHECK(!shape->Contains(point));
  BOOST_CHECK_EQUAL(shape->DistFromOutside(point, direction, 3, TGeoShape::Big()), TGeoShape::Big());
  BOOST_CHECK_EQUAL(shape->Safety(point, kFALSE), TGeoShape::Big());
}

// ---------------------------------------------------------------------------------------------
// BVH == Loop, bit for bit
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(ContainsMatchesLoopOnAGrid)
{
  Grid grid = makeGrid("contains_grid", 6, 3., 1.);
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double reach = 1.3 * gridReach(grid);
  Rng rng(20260823);
  int inside = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const double point[3] = {rng.uniform(-reach, reach), rng.uniform(-reach, reach), rng.uniform(-reach, reach)};
    clearNodeIndices(grid.assembly);
    const bool fromBVH = shape->Contains(point);
    const int bvhNode = grid.assembly->GetCurrentNodeIndex();
    clearNodeIndices(grid.assembly);
    const bool fromLoop = shape->Contains_Loop(point);
    const int loopNode = grid.assembly->GetCurrentNodeIndex();
    BOOST_REQUIRE_EQUAL(fromBVH, fromLoop);
    BOOST_REQUIRE_EQUAL(bvhNode, loopNode);
    inside += fromBVH ? 1 : 0;
  }
  // the corpus has to actually exercise both verdicts
  BOOST_CHECK_GT(inside, 100);
  BOOST_CHECK_LT(inside, 19900);
}

BOOST_AUTO_TEST_CASE(DistFromOutsideMatchesLoopOnAGrid)
{
  Grid grid = makeGrid("dist_grid", 6, 3., 1.);
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double start = 3. * gridReach(grid);
  Rng rng(777);
  int hits = 0;
  for (int trial = 0; trial < 5000; ++trial) {
    double direction[3];
    rng.direction(direction);
    const double origin[3] = {-start * direction[0], -start * direction[1], -start * direction[2]};
    // aim back through a random point in the grid volume
    const double target[3] = {rng.uniform(-gridReach(grid), gridReach(grid)),
                              rng.uniform(-gridReach(grid), gridReach(grid)),
                              rng.uniform(-gridReach(grid), gridReach(grid))};
    double aim[3] = {target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]};
    const double norm = std::sqrt(aim[0] * aim[0] + aim[1] * aim[1] + aim[2] * aim[2]);
    for (int axis = 0; axis < 3; ++axis) {
      aim[axis] /= norm;
    }
    clearNodeIndices(grid.assembly);
    const double fromBVH = shape->DistFromOutside(origin, aim, 3, TGeoShape::Big());
    const int bvhNode = grid.assembly->GetNextNodeIndex();
    clearNodeIndices(grid.assembly);
    const double fromLoop = shape->DistFromOutside_Loop(origin, aim, TGeoShape::Big());
    const int loopNode = grid.assembly->GetNextNodeIndex();
    BOOST_REQUIRE_EQUAL(fromBVH, fromLoop); // exact: both minimise the same per-daughter numbers
    BOOST_REQUIRE_EQUAL(bvhNode, loopNode);
    hits += fromBVH < TGeoShape::Big() ? 1 : 0;
  }
  BOOST_CHECK_GT(hits, 1000);
}

BOOST_AUTO_TEST_CASE(DistFromOutsideRespectsTheStepBound)
{
  Grid grid = makeGrid("step_grid", 5, 3., 1.); // odd count, so a cell sits on the axis
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double origin[3] = {-50., 0., 0.};
  const double direction[3] = {1., 0., 0.};
  const double unbounded = shape->DistFromOutside(origin, direction, 3, TGeoShape::Big());
  BOOST_REQUIRE_LT(unbounded, TGeoShape::Big());
  // a bound just short of the crossing must hide it, one just past must not
  BOOST_CHECK_EQUAL(shape->DistFromOutside(origin, direction, 3, unbounded * 0.5), TGeoShape::Big());
  BOOST_CHECK_EQUAL(shape->DistFromOutside(origin, direction, 3, unbounded * 1.5), unbounded);
  BOOST_CHECK_EQUAL(shape->DistFromOutside_Loop(origin, direction, unbounded * 0.5), TGeoShape::Big());
}

BOOST_AUTO_TEST_CASE(SafetyMatchesLoopOnAGrid)
{
  Grid grid = makeGrid("safety_grid", 6, 3., 1.);
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double reach = 1.5 * gridReach(grid);
  Rng rng(4242);
  int positive = 0;
  for (int trial = 0; trial < 4000; ++trial) {
    const double point[3] = {rng.uniform(-reach, reach), rng.uniform(-reach, reach), rng.uniform(-reach, reach)};
    const double fromBVH = shape->Safety(point, kFALSE);
    const double fromLoop = shape->Safety_Loop(point, kFALSE);
    BOOST_REQUIRE_EQUAL(fromBVH, fromLoop); // exact: the traversal prunes only on a lower bound
    positive += fromBVH > 0. ? 1 : 0;
  }
  BOOST_CHECK_GT(positive, 100);
}

// ---------------------------------------------------------------------------------------------
// Agreement with ROOT
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(ContainsAgreesWithRootOnAClosedGeometry)
{
  Grid grid = makeGrid("root_contains", 6, 3., 1.);
  grid.manager->CloseGeometry();
  BOOST_REQUIRE(grid.assembly->GetVoxels() != nullptr); // ROOT's accelerated path, not the linear one
  auto* rootShape = static_cast<TGeoShapeAssembly*>(grid.assembly->GetShape());
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double reach = 1.3 * gridReach(grid);
  Rng rng(31337);
  for (int trial = 0; trial < 10000; ++trial) {
    const double point[3] = {rng.uniform(-reach, reach), rng.uniform(-reach, reach), rng.uniform(-reach, reach)};
    const bool fromRoot = rootShape->Contains(point);
    const int rootNode = grid.assembly->GetCurrentNodeIndex();
    const bool fromBVH = shape->Contains(point);
    BOOST_REQUIRE_EQUAL(fromRoot, fromBVH);
    if (fromBVH) {
      BOOST_REQUIRE_EQUAL(rootNode, grid.assembly->GetCurrentNodeIndex()); // the same daughter, not just a daughter
    }
  }
}

/// ROOT's TGeoShapeAssembly::Safety prunes daughters on the *Euclidean* gap to their bounding
/// boxes while TGeoBBox::Safety answers the *axis-max* gap, which is smaller -- so ROOT discards
/// daughters that would have answered less and returns more than the minimum over its own
/// daughters. This class prunes on the axis-max gap and returns the minimum. The requirement is
/// therefore one-sided: never more than ROOT, and always exactly the loop.
BOOST_AUTO_TEST_CASE(SafetyIsNeverLargerThanRoot)
{
  Grid grid = makeGrid("root_safety", 5, 3., 1.);
  grid.manager->CloseGeometry();
  auto* rootShape = static_cast<TGeoShapeAssembly*>(grid.assembly->GetShape());
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double reach = 1.5 * gridReach(grid);
  Rng rng(99);
  int rootTooLarge = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const double point[3] = {rng.uniform(-reach, reach), rng.uniform(-reach, reach), rng.uniform(-reach, reach)};
    const double ours = shape->Safety(point, kFALSE);
    const double theirs = rootShape->Safety(point, kFALSE);
    BOOST_REQUIRE_EQUAL(ours, shape->Safety_Loop(point, kFALSE));
    BOOST_REQUIRE_LE(ours, theirs);
    rootTooLarge += theirs > ours ? 1 : 0;
  }
  BOOST_TEST_MESSAGE("ROOT returned more than the daughter minimum on " << rootTooLarge << " of 2000 points");
}

/// ROOT's DistFromOutside gives up on a point outside the assembly bounding box when the volume is
/// voxelized. This pins the *direction* of the disagreement:
/// ROOT is allowed to be right or to return Big(), never to return a different finite answer. It
/// keeps passing if ROOT is fixed upstream.
BOOST_AUTO_TEST_CASE(DistFromOutsideIsNeverWorseThanRoot)
{
  Grid grid = makeGrid("root_dist", 6, 3., 1.);
  grid.manager->CloseGeometry();
  BOOST_REQUIRE(grid.assembly->GetVoxels() != nullptr);
  auto* rootShape = static_cast<TGeoShapeAssembly*>(grid.assembly->GetShape());
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double start = 3. * gridReach(grid);
  Rng rng(2024);
  int weFound = 0;
  int rootGaveUp = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    double direction[3];
    rng.direction(direction);
    const double origin[3] = {-start * direction[0], -start * direction[1], -start * direction[2]};
    const double target[3] = {rng.uniform(-gridReach(grid), gridReach(grid)),
                              rng.uniform(-gridReach(grid), gridReach(grid)),
                              rng.uniform(-gridReach(grid), gridReach(grid))};
    double aim[3] = {target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]};
    const double norm = std::sqrt(aim[0] * aim[0] + aim[1] * aim[1] + aim[2] * aim[2]);
    for (int axis = 0; axis < 3; ++axis) {
      aim[axis] /= norm;
    }
    const double ours = shape->DistFromOutside(origin, aim, 3, TGeoShape::Big());
    const double theirs = rootShape->DistFromOutside(origin, aim, 3, TGeoShape::Big());
    BOOST_REQUIRE_EQUAL(ours, shape->DistFromOutside_Loop(origin, aim, TGeoShape::Big()));
    if (theirs < TGeoShape::Big()) {
      BOOST_REQUIRE_EQUAL(theirs, ours);
    } else {
      ++rootGaveUp;
    }
    weFound += ours < TGeoShape::Big() ? 1 : 0;
  }
  BOOST_CHECK_GT(weFound, 500);
  BOOST_TEST_MESSAGE("ROOT returned Big() on " << rootGaveUp << " of 2000 rays this class answered");
}

// ---------------------------------------------------------------------------------------------
// Overlaps, nesting, rotations
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(OverlappingDaughtersResolveToTheLowestIndex)
{
  auto* manager = new TGeoManager("overlap_asm", "overlap_asm");
  auto* medium = vacuum();
  auto* world = manager->MakeBox("WORLD", medium, 50., 50., 50.);
  auto* assembly = new TGeoVolumeAssembly("OVERLAP");
  // five boxes each shifted by half their width: every interior point sits in two of them
  for (int index = 0; index < 5; ++index) {
    auto* box = manager->MakeBox(Form("ov_%d", index), medium, 2., 2., 2.);
    assembly->AddNode(box, index, new TGeoTranslation(2. * index, 0., 0.));
  }
  world->AddNode(assembly, 1, new TGeoTranslation(0., 0., 0.));
  manager->SetTopVolume(world);
  auto* shape = new O2BVHAssembly(assembly);
  Rng rng(5);
  int overlaps = 0;
  for (int trial = 0; trial < 5000; ++trial) {
    const double point[3] = {rng.uniform(-4., 12.), rng.uniform(-3., 3.), rng.uniform(-3., 3.)};
    clearNodeIndices(assembly);
    const bool fromBVH = shape->Contains(point);
    const int bvhNode = assembly->GetCurrentNodeIndex();
    clearNodeIndices(assembly);
    const bool fromLoop = shape->Contains_Loop(point);
    const int loopNode = assembly->GetCurrentNodeIndex();
    BOOST_REQUIRE_EQUAL(fromBVH, fromLoop);
    BOOST_REQUIRE_EQUAL(bvhNode, loopNode);
    if (fromBVH) {
      int count = 0;
      double local[3];
      for (int index = 0; index < assembly->GetNdaughters(); ++index) {
        assembly->GetNode(index)->MasterToLocal(point, local);
        count += assembly->GetNode(index)->GetVolume()->GetShape()->Contains(local) ? 1 : 0;
      }
      overlaps += count > 1 ? 1 : 0;
    }
  }
  BOOST_CHECK_GT(overlaps, 100); // the corpus really does have shared points
}

BOOST_AUTO_TEST_CASE(NestedAssembliesAgreeWithTheLoop)
{
  auto* manager = new TGeoManager("nested_asm", "nested_asm");
  auto* medium = vacuum();
  auto* world = manager->MakeBox("WORLD", medium, 100., 100., 100.);
  auto* outer = new TGeoVolumeAssembly("OUTER");
  for (int block = 0; block < 6; ++block) {
    auto* inner = new TGeoVolumeAssembly(Form("INNER_%d", block));
    for (int cell = 0; cell < 8; ++cell) {
      auto* box = manager->MakeBox(Form("n_%d_%d", block, cell), medium, 1., 1., 1.);
      inner->AddNode(box, cell, new TGeoTranslation(2.5 * cell, 0., 0.));
    }
    auto* rotation = new TGeoRotation(Form("rot_%d", block), 13. * block, 7. * block, 5. * block);
    outer->AddNode(inner, block, new TGeoCombiTrans(0., 6. * block, 0., rotation));
  }
  world->AddNode(outer, 1, new TGeoTranslation(0., 0., 0.));
  manager->SetTopVolume(world);
  auto* shape = new O2BVHAssembly(outer);
  BOOST_CHECK_EQUAL(shape->GetNbuilt(), 6);
  Rng rng(606);
  int inside = 0;
  int hits = 0;
  for (int trial = 0; trial < 4000; ++trial) {
    const double point[3] = {rng.uniform(-5., 25.), rng.uniform(-5., 35.), rng.uniform(-5., 5.)};
    clearNodeIndices(outer);
    const bool fromBVH = shape->Contains(point);
    const int bvhNode = outer->GetCurrentNodeIndex();
    clearNodeIndices(outer);
    const bool fromLoop = shape->Contains_Loop(point);
    BOOST_REQUIRE_EQUAL(fromBVH, fromLoop);
    BOOST_REQUIRE_EQUAL(bvhNode, outer->GetCurrentNodeIndex());
    inside += fromBVH ? 1 : 0;

    double direction[3];
    rng.direction(direction);
    const double origin[3] = {point[0] - 200. * direction[0], point[1] - 200. * direction[1],
                              point[2] - 200. * direction[2]};
    const double distanceBVH = shape->DistFromOutside(origin, direction, 3, TGeoShape::Big());
    const double distanceLoop = shape->DistFromOutside_Loop(origin, direction, TGeoShape::Big());
    BOOST_REQUIRE_EQUAL(distanceBVH, distanceLoop);
    hits += distanceBVH < TGeoShape::Big() ? 1 : 0;

    BOOST_REQUIRE_EQUAL(shape->Safety(point, kFALSE), shape->Safety_Loop(point, kFALSE));
  }
  BOOST_CHECK_GT(inside, 50);
  BOOST_CHECK_GT(hits, 500);
}

BOOST_AUTO_TEST_CASE(RotatedDaughtersAgreeWithTheLoop)
{
  auto* manager = new TGeoManager("rotated_asm", "rotated_asm");
  auto* medium = vacuum();
  auto* world = manager->MakeBox("WORLD", medium, 100., 100., 100.);
  auto* assembly = new TGeoVolumeAssembly("ROTATED");
  for (int index = 0; index < 40; ++index) {
    auto* box = manager->MakeBox(Form("r_%d", index), medium, 3., 0.5, 2.);
    auto* rotation = new TGeoRotation(Form("rr_%d", index), 9. * index, 4. * index, 17. * index);
    assembly->AddNode(box, index,
                      new TGeoCombiTrans(8. * std::cos(0.31 * index), 8. * std::sin(0.31 * index), 0.7 * index - 14.,
                                         rotation));
  }
  world->AddNode(assembly, 1, new TGeoTranslation(0., 0., 0.));
  manager->SetTopVolume(world);
  auto* shape = new O2BVHAssembly(assembly);
  Rng rng(818);
  for (int trial = 0; trial < 4000; ++trial) {
    const double point[3] = {rng.uniform(-15., 15.), rng.uniform(-15., 15.), rng.uniform(-20., 20.)};
    BOOST_REQUIRE_EQUAL(shape->Contains(point), shape->Contains_Loop(point));
    BOOST_REQUIRE_EQUAL(shape->Safety(point, kFALSE), shape->Safety_Loop(point, kFALSE));
    double direction[3];
    rng.direction(direction);
    BOOST_REQUIRE_EQUAL(shape->DistFromOutside(point, direction, 3, TGeoShape::Big()),
                        shape->DistFromOutside_Loop(point, direction, TGeoShape::Big()));
  }
}

// ---------------------------------------------------------------------------------------------
// Edge cases the tolerance discipline exists for
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(PointsOnSharedFacesAgreeWithTheLoop)
{
  // touching cells: pitch equals the box width, so consecutive cells share a face exactly
  Grid grid = makeGrid("faces_grid", 5, 2., 1.);
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double first = -0.5 * grid.pitch * (grid.count - 1);
  int checked = 0;
  for (int ix = 0; ix < grid.count; ++ix) {
    for (int iy = 0; iy < grid.count; ++iy) {
      for (int iz = 0; iz < grid.count; ++iz) {
        // the +x face of cell (ix,iy,iz), which is the -x face of its neighbour
        const double point[3] = {first + grid.pitch * ix + grid.halfBox, first + grid.pitch * iy,
                                 first + grid.pitch * iz};
        clearNodeIndices(grid.assembly);
        const bool fromBVH = shape->Contains(point);
        const int bvhNode = grid.assembly->GetCurrentNodeIndex();
        clearNodeIndices(grid.assembly);
        BOOST_REQUIRE_EQUAL(fromBVH, shape->Contains_Loop(point));
        BOOST_REQUIRE_EQUAL(bvhNode, grid.assembly->GetCurrentNodeIndex());
        BOOST_REQUIRE_EQUAL(shape->Safety(point, kFALSE), shape->Safety_Loop(point, kFALSE));
        ++checked;
      }
    }
  }
  BOOST_CHECK_EQUAL(checked, 125);
}

BOOST_AUTO_TEST_CASE(RaysAlongASeamAgreeWithTheLoop)
{
  Grid grid = makeGrid("seam_grid", 5, 2., 1.);
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double first = -0.5 * grid.pitch * (grid.count - 1);
  const double start = 4. * gridReach(grid);
  int checked = 0;
  for (int iy = 0; iy < grid.count; ++iy) {
    for (int iz = 0; iz < grid.count; ++iz) {
      for (int offset = -1; offset <= 1; ++offset) {
        // a ray running exactly along the plane where two rows of cells touch
        const double y = first + grid.pitch * iy + offset * grid.halfBox;
        const double origin[3] = {-start, y, first + grid.pitch * iz};
        const double direction[3] = {1., 0., 0.};
        clearNodeIndices(grid.assembly);
        const double fromBVH = shape->DistFromOutside(origin, direction, 3, TGeoShape::Big());
        const int bvhNode = grid.assembly->GetNextNodeIndex();
        clearNodeIndices(grid.assembly);
        BOOST_REQUIRE_EQUAL(fromBVH, shape->DistFromOutside_Loop(origin, direction, TGeoShape::Big()));
        BOOST_REQUIRE_EQUAL(bvhNode, grid.assembly->GetNextNodeIndex());
        ++checked;
      }
    }
  }
  BOOST_CHECK_EQUAL(checked, 75);
}

BOOST_AUTO_TEST_CASE(RaysAlongTheCoordinateAxesAgreeWithTheLoop)
{
  Grid grid = makeGrid("axis_grid", 5, 3., 1.);
  auto* shape = new O2BVHAssembly(grid.assembly);
  const double start = 4. * gridReach(grid);
  const double first = -0.5 * grid.pitch * (grid.count - 1);
  for (int axis = 0; axis < 3; ++axis) {
    for (int step = 0; step < grid.count; ++step) {
      double origin[3] = {0., 0., 0.};
      double direction[3] = {0., 0., 0.};
      origin[axis] = -start;
      direction[axis] = 1.;
      origin[(axis + 1) % 3] = first + grid.pitch * step;
      const double fromBVH = shape->DistFromOutside(origin, direction, 3, TGeoShape::Big());
      BOOST_REQUIRE_EQUAL(fromBVH, shape->DistFromOutside_Loop(origin, direction, TGeoShape::Big()));
      BOOST_REQUIRE_LT(fromBVH, TGeoShape::Big());
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Lifecycle: lazy rebuild, shape swap, navigation, I/O
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(AddingADaughterRebuildsLazily)
{
  auto* manager = new TGeoManager("lazy_asm", "lazy_asm");
  auto* medium = vacuum();
  auto* world = manager->MakeBox("WORLD", medium, 50., 50., 50.);
  auto* assembly = new TGeoVolumeAssembly("LAZY");
  auto* firstBox = manager->MakeBox("lz_0", medium, 1., 1., 1.);
  assembly->AddNode(firstBox, 0, new TGeoTranslation(0., 0., 0.));
  world->AddNode(assembly, 1, new TGeoTranslation(0., 0., 0.));
  manager->SetTopVolume(world);

  auto* shape = new O2BVHAssembly(assembly);
  assembly->SetShape(shape);
  const double newPoint[3] = {10., 0., 0.};
  BOOST_CHECK(!shape->Contains(newPoint));

  auto* secondBox = manager->MakeBox("lz_1", medium, 1., 1., 1.);
  assembly->AddNode(secondBox, 1, new TGeoTranslation(10., 0., 0.));
  // AddNode invalidated the base bounding box; the BVH notices the new daughter count by itself
  BOOST_CHECK(shape->Contains(newPoint));
  BOOST_CHECK_EQUAL(assembly->GetCurrentNodeIndex(), 1);
  BOOST_CHECK_EQUAL(shape->GetNbuilt(), 2);
}

BOOST_AUTO_TEST_CASE(MakeBVHAssemblySwapsTheShapeAndKeepsTheVoxels)
{
  Grid grid = makeGrid("swap_asm", 5, 3., 1.);
  grid.manager->CloseGeometry();
  BOOST_REQUIRE(grid.assembly->GetVoxels() != nullptr);
  auto* shape = O2BVHAssembly::MakeBVHAssembly(grid.assembly);
  BOOST_REQUIRE(shape != nullptr);
  BOOST_CHECK_EQUAL(grid.assembly->GetShape(), static_cast<TGeoShape*>(shape));
  BOOST_CHECK(grid.assembly->IsAssembly());
  BOOST_CHECK(shape->IsAssembly());
  BOOST_CHECK_EQUAL(shape->GetNbuilt(), 125);
  // the finder stays by default: TGeoNavigator::SearchNode reads it once it is inside the
  // assembly, and dropping it turns point location into a linear walk
  BOOST_CHECK(grid.assembly->GetVoxels() != nullptr);
  BOOST_CHECK(O2BVHAssembly::MakeBVHAssembly(nullptr) == nullptr);
}

BOOST_AUTO_TEST_CASE(NavigationFindsTheSameLeafBeforeAndAfterTheSwap)
{
  Grid grid = makeGrid("nav_asm", 5, 3., 1.);
  grid.manager->CloseGeometry();
  const double reach = 1.2 * gridReach(grid);
  Rng rng(1234);
  std::vector<double> points;
  std::vector<std::string> paths;
  for (int trial = 0; trial < 3000; ++trial) {
    const double point[3] = {rng.uniform(-reach, reach), rng.uniform(-reach, reach), rng.uniform(-reach, reach)};
    grid.manager->FindNode(point[0], point[1], point[2]);
    points.insert(points.end(), {point[0], point[1], point[2]});
    paths.emplace_back(grid.manager->GetPath());
  }
  O2BVHAssembly::MakeBVHAssembly(grid.assembly);
  int deep = 0;
  for (size_t trial = 0; trial < paths.size(); ++trial) {
    grid.manager->FindNode(points[3 * trial], points[3 * trial + 1], points[3 * trial + 2]);
    BOOST_REQUIRE_EQUAL(paths[trial], std::string(grid.manager->GetPath()));
    deep += paths[trial].find("/cell_") != std::string::npos ? 1 : 0;
  }
  BOOST_CHECK_GT(deep, 100); // the corpus really does reach the leaves through the assembly
}

BOOST_AUTO_TEST_CASE(TransportCrossesTheSameLeavesAsRoot)
{
  Grid reference = makeGrid("transport_root", 5, 3., 1.);
  reference.manager->CloseGeometry();
  const double start = 3. * gridReach(reference);
  Rng rng(24680);
  std::vector<double> origins;
  std::vector<double> directions;
  std::vector<std::vector<std::string>> rootPaths;
  for (int ray = 0; ray < 200; ++ray) {
    double direction[3];
    rng.direction(direction);
    const double origin[3] = {-start * direction[0], -start * direction[1], -start * direction[2]};
    origins.insert(origins.end(), {origin[0], origin[1], origin[2]});
    directions.insert(directions.end(), {direction[0], direction[1], direction[2]});
    reference.manager->InitTrack(origin, direction);
    std::vector<std::string> path;
    int guard = 0;
    while (!reference.manager->IsOutside() && guard++ < 500) {
      reference.manager->FindNextBoundaryAndStep(1.e10);
      path.emplace_back(reference.manager->GetPath());
    }
    rootPaths.push_back(path);
  }

  O2BVHAssembly::MakeBVHAssembly(reference.assembly);
  int crossings = 0;
  for (int ray = 0; ray < 200; ++ray) {
    reference.manager->InitTrack(&origins[3 * ray], &directions[3 * ray]);
    std::vector<std::string> path;
    int guard = 0;
    while (!reference.manager->IsOutside() && guard++ < 500) {
      reference.manager->FindNextBoundaryAndStep(1.e10);
      path.emplace_back(reference.manager->GetPath());
    }
    // this class only ever finds *more* than ROOT (section 4 of the stream document), so the
    // requirement is that everything ROOT saw is still seen, in order
    BOOST_REQUIRE_GE(path.size(), rootPaths[ray].size());
    for (const auto& step : rootPaths[ray]) {
      crossings += step.find("/cell_") != std::string::npos ? 1 : 0;
    }
    if (path.size() == rootPaths[ray].size()) {
      BOOST_REQUIRE(path == rootPaths[ray]);
    }
  }
  BOOST_CHECK_GT(crossings, 100);
}

BOOST_AUTO_TEST_CASE(SurvivesAGeometryRoundTrip)
{
  const std::string file = "testBVHAssembly_roundtrip.root";
  Grid grid = makeGrid("io_asm", 4, 3., 1.);
  grid.manager->CloseGeometry();
  O2BVHAssembly::MakeBVHAssembly(grid.assembly);
  const double reach = 1.2 * gridReach(grid);
  Rng rng(1111);
  std::vector<double> points;
  std::vector<int> nodes;
  auto* shape = static_cast<O2BVHAssembly*>(grid.assembly->GetShape());
  for (int trial = 0; trial < 2000; ++trial) {
    const double point[3] = {rng.uniform(-reach, reach), rng.uniform(-reach, reach), rng.uniform(-reach, reach)};
    points.insert(points.end(), {point[0], point[1], point[2]});
    clearNodeIndices(grid.assembly);
    shape->Contains(point);
    nodes.push_back(grid.assembly->GetCurrentNodeIndex());
  }
  grid.manager->Export(file.c_str());

  auto* reloaded = TGeoManager::Import(file.c_str());
  BOOST_REQUIRE(reloaded != nullptr);
  auto* reloadedAssembly = dynamic_cast<TGeoVolumeAssembly*>(reloaded->GetTopVolume()->GetNode(0)->GetVolume());
  BOOST_REQUIRE(reloadedAssembly != nullptr);
  auto* reloadedShape = dynamic_cast<O2BVHAssembly*>(reloadedAssembly->GetShape());
  BOOST_REQUIRE(reloadedShape != nullptr); // the shape survived streaming as itself
  for (size_t trial = 0; trial < nodes.size(); ++trial) {
    clearNodeIndices(reloadedAssembly);
    reloadedShape->Contains(&points[3 * trial]);
    BOOST_REQUIRE_EQUAL(nodes[trial], reloadedAssembly->GetCurrentNodeIndex());
  }
  BOOST_CHECK_EQUAL(reloadedShape->GetNbuilt(), 64); // rebuilt lazily on the first query
  std::error_code ignored;
  std::filesystem::remove(file, ignored);
}
