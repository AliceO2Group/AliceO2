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

/// \file O2OverlapCheck.cxx
/// \brief An overlap census that asks the shapes: every sampled point is verified to lie on its solid's boundary,
/// and the depth, not containment, separates touching from interpenetrating pairs.

#include "CADSupport/O2OverlapCheck.h"

#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace o2
{
namespace cad
{

const char* OverlapVerdictName(OverlapVerdict verdict)
{
  switch (verdict) {
    case OverlapVerdict::Disjoint:
      return "disjoint";
    case OverlapVerdict::Touching:
      return "touching";
    case OverlapVerdict::Interpenetrating:
      return "INTERPENETRATING";
    case OverlapVerdict::Contained:
      return "CONTAINED";
  }
  return "unknown";
}

namespace
{

/// The master-frame axis-aligned box of a shape's local bounding box under \a matrix, inflated by
/// \a pad. Conservative for a rotation because it takes the box of the eight transformed corners.
struct MasterBox {
  double lower[3] = {0., 0., 0.};
  double upper[3] = {0., 0., 0.};
  bool valid = false;
};

MasterBox masterBox(const TGeoShape* shape, const TGeoMatrix* matrix, double pad)
{
  MasterBox box;
  const auto* boundingBox = dynamic_cast<const TGeoBBox*>(shape);
  if (boundingBox == nullptr) {
    return box;
  }
  const double* origin = boundingBox->GetOrigin();
  const double halfLengths[3] = {boundingBox->GetDX(), boundingBox->GetDY(), boundingBox->GetDZ()};
  for (int dimension = 0; dimension < 3; ++dimension) {
    box.lower[dimension] = std::numeric_limits<double>::max();
    box.upper[dimension] = -std::numeric_limits<double>::max();
  }
  for (int corner = 0; corner < 8; ++corner) {
    const double local[3] = {origin[0] + ((corner & 1) ? halfLengths[0] : -halfLengths[0]),
                             origin[1] + ((corner & 2) ? halfLengths[1] : -halfLengths[1]),
                             origin[2] + ((corner & 4) ? halfLengths[2] : -halfLengths[2])};
    double master[3] = {0., 0., 0.};
    matrix->LocalToMaster(local, master);
    for (int dimension = 0; dimension < 3; ++dimension) {
      box.lower[dimension] = std::min(box.lower[dimension], master[dimension]);
      box.upper[dimension] = std::max(box.upper[dimension], master[dimension]);
    }
  }
  for (int dimension = 0; dimension < 3; ++dimension) {
    box.lower[dimension] -= pad;
    box.upper[dimension] += pad;
  }
  box.valid = true;
  return box;
}

bool boxesOverlap(const MasterBox& first, const MasterBox& second)
{
  if (!first.valid || !second.valid) {
    return true; // no box means no rejection; test the pair
  }
  for (int dimension = 0; dimension < 3; ++dimension) {
    if (first.upper[dimension] < second.lower[dimension] || second.upper[dimension] < first.lower[dimension]) {
      return false;
    }
  }
  return true;
}

/// Radical-inverse (Halton) coordinate; deterministic, so two runs differ only if the geometry does.
inline double halton(unsigned int index, unsigned int base)
{
  double result = 0.;
  double fraction = 1.;
  while (index > 0) {
    fraction /= base;
    result += fraction * (index % base);
    index /= base;
  }
  return result;
}

} // namespace

int SampleBoundaryPoints(const TGeoShape* shape, int npoints, double residualTolerance,
                         std::vector<double>& points, int& rejected, double& worstResidual,
                         bool* usedPointsOnSegments)
{
  points.clear();
  rejected = 0;
  worstResidual = 0.;
  if (usedPointsOnSegments != nullptr) {
    *usedPointsOnSegments = false;
  }
  if (shape == nullptr || npoints <= 0) {
    return 0;
  }

  int meshVertices = 0;
  int meshSegments = 0;
  int meshPolygons = 0;
  shape->GetMeshNumbers(meshVertices, meshSegments, meshPolygons);

  // TGeoChecker::MakeCheckOverlap's choice: a shape that declines to sample still has display vertices
  const int capacity = std::max(npoints, meshVertices);
  std::vector<double> raw(3 * static_cast<size_t>(std::max(capacity, 1)), 0.);
  int rawCount = 0;
  if (shape->GetPointsOnSegments(npoints, raw.data())) {
    rawCount = npoints;
    if (usedPointsOnSegments != nullptr) {
      *usedPointsOnSegments = true;
    }
  } else {
    if (meshVertices <= 0) {
      return 0;
    }
    shape->SetPoints(raw.data());
    rawCount = meshVertices;
  }

  points.reserve(3 * static_cast<size_t>(rawCount));
  for (int index = 0; index < rawCount; ++index) {
    const double* candidate = &raw[3 * static_cast<size_t>(index)];
    // Safety() is a lower bound on the distance to the boundary, so a large value is a proof that
    // the point is *not* on it. That is the direction this filter needs.
    const double residual = shape->Safety(candidate, shape->Contains(candidate));
    if (!(residual <= residualTolerance)) {
      rejected++;
      continue;
    }
    worstResidual = std::max(worstResidual, residual);
    points.push_back(candidate[0]);
    points.push_back(candidate[1]);
    points.push_back(candidate[2]);
  }
  return static_cast<int>(points.size() / 3);
}

namespace
{

/// One direction of the pair test: every accepted boundary point of \a points (in \a matFrom's
/// local frame) against \a target.
struct DirectionResult {
  int contained = 0;
  int deep = 0;
  double maxDepth = 0.;
  double deepestMaster[3] = {0., 0., 0.};
  double minSeparation = std::numeric_limits<double>::max();
};

DirectionResult probeDirection(const std::vector<double>& points, const TGeoMatrix* matFrom,
                               const TGeoShape* target, const TGeoMatrix* matTo, double depthTolerance)
{
  DirectionResult result;
  const size_t count = points.size() / 3;
  for (size_t index = 0; index < count; ++index) {
    double master[3] = {0., 0., 0.};
    double local[3] = {0., 0., 0.};
    matFrom->LocalToMaster(&points[3 * index], master);
    matTo->MasterToLocal(master, local);
    if (target->Contains(local)) {
      result.contained++;
      const double depth = target->Safety(local, kTRUE);
      if (depth > depthTolerance) {
        result.deep++;
      }
      if (depth > result.maxDepth) {
        result.maxDepth = depth;
        std::memcpy(result.deepestMaster, master, 3 * sizeof(double));
      }
    } else {
      result.minSeparation = std::min(result.minSeparation, target->Safety(local, kFALSE));
    }
  }
  return result;
}

/// Probe a sampled pair both ways and set its counts, depth, deepest point and verdict.
OverlapPair assemblePair(const std::string& nameA, const std::vector<double>& pointsA, const TGeoShape* shapeA,
                         const TGeoMatrix* matA, const std::string& nameB, const std::vector<double>& pointsB,
                         const TGeoShape* shapeB, const TGeoMatrix* matB, const OverlapOptions& options)
{
  OverlapPair pair;
  pair.nameA = nameA;
  pair.nameB = nameB;
  pair.sampledA = static_cast<int>(pointsA.size() / 3);
  pair.sampledB = static_cast<int>(pointsB.size() / 3);

  const DirectionResult aInB = probeDirection(pointsA, matA, shapeB, matB, options.depthTolerance);
  const DirectionResult bInA = probeDirection(pointsB, matB, shapeA, matA, options.depthTolerance);

  pair.pointsAInsideB = aInB.contained;
  pair.pointsBInsideA = bInA.contained;
  pair.deepPointsAInsideB = aInB.deep;
  pair.deepPointsBInsideA = bInA.deep;

  if (aInB.maxDepth >= bInA.maxDepth) {
    pair.depthCm = aInB.maxDepth;
    std::copy(aInB.deepestMaster, aInB.deepestMaster + 3, pair.deepestPoint.begin());
    pair.deepestPointFrom = nameA;
  } else {
    pair.depthCm = bInA.maxDepth;
    std::copy(bInA.deepestMaster, bInA.deepestMaster + 3, pair.deepestPoint.begin());
    pair.deepestPointFrom = nameB;
  }

  // Containment: every boundary point of one solid is inside the other, and none of them is merely
  // on its boundary. Legal only as a declared mother/daughter, which a flat conversion never emits.
  const bool allAInside = pair.sampledA > 0 && aInB.contained == pair.sampledA && aInB.deep == pair.sampledA;
  const bool allBInside = pair.sampledB > 0 && bInA.contained == pair.sampledB && bInA.deep == pair.sampledB;

  if (allAInside || allBInside) {
    pair.verdict = OverlapVerdict::Contained;
  } else if (aInB.deep > 0 || bInA.deep > 0) {
    pair.verdict = OverlapVerdict::Interpenetrating;
  } else if (aInB.contained > 0 || bInA.contained > 0) {
    pair.verdict = OverlapVerdict::Touching;
  } else {
    pair.verdict = OverlapVerdict::Disjoint;
    const double separation = std::min(aInB.minSeparation, bInA.minSeparation);
    if (separation < std::numeric_limits<double>::max()) {
      pair.separationCm = separation;
    }
  }
  return pair;
}

/// Monte-Carlo estimate of the volume two placed solids share, into \a pair's shared-volume fields.
void estimateSharedVolume(const TGeoShape* shapeA, const TGeoMatrix* matA, const TGeoShape* shapeB,
                          const TGeoMatrix* matB, int samples, OverlapPair& pair)
{
  const MasterBox boxA = masterBox(shapeA, matA, 0.);
  const MasterBox boxB = masterBox(shapeB, matB, 0.);
  if (!boxA.valid || !boxB.valid) {
    return;
  }
  double lower[3];
  double upper[3];
  double boxVolume = 1.;
  for (int dimension = 0; dimension < 3; ++dimension) {
    lower[dimension] = std::max(boxA.lower[dimension], boxB.lower[dimension]);
    upper[dimension] = std::min(boxA.upper[dimension], boxB.upper[dimension]);
    boxVolume *= std::max(0., upper[dimension] - lower[dimension]);
  }
  if (!(boxVolume > 0.)) {
    return;
  }
  int hits = 0;
  for (int sample = 0; sample < samples; ++sample) {
    const double master[3] = {lower[0] + (upper[0] - lower[0]) * halton(sample + 1, 2),
                              lower[1] + (upper[1] - lower[1]) * halton(sample + 1, 3),
                              lower[2] + (upper[2] - lower[2]) * halton(sample + 1, 5)};
    double local[3];
    matA->MasterToLocal(master, local);
    if (!shapeA->Contains(local)) {
      continue;
    }
    matB->MasterToLocal(master, local);
    if (shapeB->Contains(local)) {
      hits++;
    }
  }
  const double fraction = double(hits) / samples;
  pair.sharedVolumeHits = hits;
  pair.sharedVolumeCm3 = fraction * boxVolume;
  pair.sharedVolumeErrCm3 = std::sqrt(std::max(1., double(hits))) / samples * boxVolume;
}

} // namespace

OverlapPair CheckPairOverlap(const TGeoShape* shapeA, const TGeoMatrix* matA, const std::string& nameA,
                             const TGeoShape* shapeB, const TGeoMatrix* matB, const std::string& nameB,
                             const OverlapOptions& options)
{
  OverlapPair pair;
  pair.nameA = nameA;
  pair.nameB = nameB;
  if (shapeA == nullptr || shapeB == nullptr || matA == nullptr || matB == nullptr) {
    return pair;
  }

  int rejectedA = 0;
  int rejectedB = 0;
  double residualA = 0.;
  double residualB = 0.;
  std::vector<double> pointsA;
  std::vector<double> pointsB;
  SampleBoundaryPoints(shapeA, options.pointsPerSolid, options.residualTolerance, pointsA, rejectedA, residualA);
  SampleBoundaryPoints(shapeB, options.pointsPerSolid, options.residualTolerance, pointsB, rejectedB, residualB);
  pair = assemblePair(nameA, pointsA, shapeA, matA, nameB, pointsB, shapeB, matB, options);
  if (options.volumeSamples > 0 &&
      (pair.verdict == OverlapVerdict::Interpenetrating || pair.verdict == OverlapVerdict::Contained)) {
    estimateSharedVolume(shapeA, matA, shapeB, matB, options.volumeSamples, pair);
  }
  return pair;
}

OverlapCensus CheckWorldOverlaps(const TGeoVolume* volume, const OverlapOptions& options)
{
  const auto startTime = std::chrono::steady_clock::now();
  OverlapCensus census;
  if (volume == nullptr) {
    return census;
  }
  const int daughters = volume->GetNdaughters();
  census.nSolids = daughters;
  census.nPairsTotal = daughters * (daughters - 1) / 2;

  std::vector<const TGeoShape*> shapes(daughters, nullptr);
  std::vector<const TGeoMatrix*> matrices(daughters, nullptr);
  std::vector<std::string> names(daughters);
  std::vector<MasterBox> boxes(daughters);
  std::vector<std::vector<double>> points(daughters);

  for (int index = 0; index < daughters; ++index) {
    TGeoNode* node = volume->GetNode(index);
    shapes[index] = node->GetVolume()->GetShape();
    matrices[index] = node->GetMatrix();
    names[index] = node->GetVolume()->GetName();
    boxes[index] = masterBox(shapes[index], matrices[index], options.padCm);

    OverlapSolidReport report;
    report.name = names[index];
    report.shapeClass = shapes[index] != nullptr ? shapes[index]->ClassName() : "none";
    report.requested = options.pointsPerSolid;
    bool usedSegments = false;
    report.accepted = SampleBoundaryPoints(shapes[index], options.pointsPerSolid, options.residualTolerance,
                                           points[index], report.rejected, report.worstResidualCm, &usedSegments);
    report.usedPointsOnSegments = usedSegments;
    census.nPointsRejected += report.rejected;
    census.worstResidualCm = std::max(census.worstResidualCm, report.worstResidualCm);
    census.solids.push_back(report);
  }

  for (int first = 0; first < daughters; ++first) {
    for (int second = first + 1; second < daughters; ++second) {
      if (!boxesOverlap(boxes[first], boxes[second])) {
        continue;
      }
      census.nPairsTested++;
      // Reuse the point sets: sampling is the expensive part and it does not depend on the partner.
      OverlapPair pair = assemblePair(names[first], points[first], shapes[first], matrices[first], names[second],
                                      points[second], shapes[second], matrices[second], options);
      switch (pair.verdict) {
        case OverlapVerdict::Disjoint:
          census.nDisjoint++;
          break;
        case OverlapVerdict::Touching:
          census.nTouching++;
          break;
        case OverlapVerdict::Interpenetrating:
          census.nInterpenetrating++;
          break;
        case OverlapVerdict::Contained:
          census.nContained++;
          break;
      }
      if (options.volumeSamples > 0 && (pair.verdict == OverlapVerdict::Interpenetrating ||
                                        pair.verdict == OverlapVerdict::Contained)) {
        estimateSharedVolume(shapes[first], matrices[first], shapes[second], matrices[second], options.volumeSamples,
                             pair);
      }
      census.pairs.push_back(pair);
    }
  }

  // extrusion: a daughter's boundary point outside its mother
  if (options.checkExtrusion && volume->GetShape() != nullptr && !volume->IsAssembly()) {
    TGeoIdentity identity;
    for (int index = 0; index < daughters; ++index) {
      OverlapPair pair;
      pair.nameA = names[index];
      pair.nameB = volume->GetName();
      pair.sampledA = static_cast<int>(points[index].size() / 3);
      const TGeoShape* mother = volume->GetShape();
      double worst = 0.;
      int outside = 0;
      double worstMaster[3] = {0., 0., 0.};
      for (size_t point = 0; point < points[index].size() / 3; ++point) {
        double master[3] = {0., 0., 0.};
        matrices[index]->LocalToMaster(&points[index][3 * point], master);
        if (!mother->Contains(master)) {
          const double depth = mother->Safety(master, kFALSE);
          if (depth > options.depthTolerance) {
            outside++;
            if (depth > worst) {
              worst = depth;
              std::memcpy(worstMaster, master, 3 * sizeof(double));
            }
          }
        }
      }
      if (outside > 0) {
        pair.verdict = OverlapVerdict::Interpenetrating;
        pair.depthCm = worst;
        pair.deepPointsAInsideB = outside;
        pair.deepestPointFrom = names[index];
        std::copy(worstMaster, worstMaster + 3, pair.deepestPoint.begin());
        census.extrusions.push_back(pair);
        census.nExtruding++;
      }
    }
  }

  census.elapsedSeconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
  return census;
}

} // namespace cad
} // namespace o2
