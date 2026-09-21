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

#ifndef ALICEO2_CADSUPPORT_O2OVERLAPCHECK_
#define ALICEO2_CADSUPPORT_O2OVERLAPCHECK_

#include <array>
#include <string>
#include <vector>

class TGeoShape;
class TGeoMatrix;
class TGeoVolume;

namespace o2
{
namespace cad
{

/// Whether two placed solids may legally coexist: disjoint and touching are legal, interpenetrating and contained are not.
enum class OverlapVerdict {
  Disjoint,         ///< no sampled boundary point of either solid lies inside the other
  Touching,         ///< boundary points coincide, but none is deeper than the depth tolerance
  Interpenetrating, ///< a boundary point of one solid lies strictly inside the other: illegal
  Contained         ///< every sampled boundary point of the smaller solid is inside the other
};

const char* OverlapVerdictName(OverlapVerdict verdict);

struct OverlapOptions {
  /// Boundary points sampled per solid. Coverage, not accuracy: every individual answer is exact,
  /// so this bounds the *false negatives* and nothing else.
  int pointsPerSolid = 20000;
  /// A containment shallower than this is a shared boundary, not an overlap. In cm.
  double depthTolerance = 1.e-6;
  /// A sampled point further than this from the boundary of the solid it was sampled from is not
  /// evidence about anything and is discarded (and counted). In cm.
  double residualTolerance = 1.e-6;
  /// Bounding-box inflation before the pairwise rejection, in cm; it decides which disjoint pairs get a separation.
  double padCm = 0.1;
  /// Monte-Carlo samples for the shared volume of an illegal pair; 0, the default, disables the estimate.
  int volumeSamples = 0;
  /// Also test every daughter against the mother it sits in (ROOT's "extrusion" case). Silently a
  /// no-op when the mother is an assembly, which has no shape to be extruded from.
  bool checkExtrusion = true;
};

/// One pair of placed solids, and everything measured about it.
struct OverlapPair {
  std::string nameA;
  std::string nameB;
  OverlapVerdict verdict = OverlapVerdict::Disjoint;

  /// The largest depth of a sampled boundary point of one solid inside the other: the verdict's evidence.
  /// A lower bound on the penetration depth, and when positive a proof that the interiors share volume.
  double depthCm = 0.;
  std::array<double, 3> deepestPoint{{0., 0., 0.}}; ///< in the master frame
  std::string deepestPointFrom;                     ///< which solid's boundary the deepest point came from

  int pointsAInsideB = 0; ///< sampled points of A found inside B at any depth
  int pointsBInsideA = 0;
  int deepPointsAInsideB = 0; ///< ... of which deeper than depthTolerance
  int deepPointsBInsideA = 0;
  int sampledA = 0; ///< accepted (on-boundary) sample counts actually used
  int sampledB = 0;

  /// Smallest sampled distance from a boundary point of one solid to the other, in cm; meaningful only when Disjoint.
  double separationCm = -1.;

  double sharedVolumeCm3 = -1.;   ///< Monte-Carlo estimate; < 0 when not measured
  double sharedVolumeErrCm3 = 0.; ///< its 1-sigma statistical error
  int sharedVolumeHits = 0;
};

/// One solid's sampling report; a shape with a poor display mesh shows here as reduced coverage.
struct OverlapSolidReport {
  std::string name;
  std::string shapeClass;
  int requested = 0;
  int accepted = 0;
  int rejected = 0;
  double worstResidualCm = 0.; ///< the largest own-boundary distance among the *accepted* points
  bool usedPointsOnSegments = false;
};

struct OverlapCensus {
  std::vector<OverlapSolidReport> solids;
  std::vector<OverlapPair> pairs; ///< only the pairs that survived the bounding-box rejection
  std::vector<OverlapPair> extrusions;

  int nSolids = 0;
  int nPairsTotal = 0;  ///< N (N - 1) / 2
  int nPairsTested = 0; ///< after the bounding-box rejection
  int nDisjoint = 0;
  int nTouching = 0;
  int nInterpenetrating = 0;
  int nContained = 0;
  int nExtruding = 0;
  int nPointsRejected = 0;
  double worstResidualCm = 0.;
  double elapsedSeconds = 0.;

  /// The one-line answer: nInterpenetrating + nContained + nExtruding.
  int illegalCount() const { return nInterpenetrating + nContained + nExtruding; }
};

/// Sample \a npoints points on \a shape's own boundary into \a points, keeping those within \a residualTolerance where Contains flips.
/// Returns the number kept; \a rejected and \a worstResidual report the filter.
int SampleBoundaryPoints(const TGeoShape* shape, int npoints, double residualTolerance,
                         std::vector<double>& points, int& rejected, double& worstResidual,
                         bool* usedPointsOnSegments = nullptr);

/// Test one placed pair. \a matA / \a matB take each shape's local frame to the common frame.
OverlapPair CheckPairOverlap(const TGeoShape* shapeA, const TGeoMatrix* matA, const std::string& nameA,
                             const TGeoShape* shapeB, const TGeoMatrix* matB, const std::string& nameB,
                             const OverlapOptions& options = OverlapOptions());

/// Census every pair of \a volume's immediate daughters, and optionally each daughter against \a volume.
OverlapCensus CheckWorldOverlaps(const TGeoVolume* volume, const OverlapOptions& options = OverlapOptions());

} // namespace cad
} // namespace o2

#endif
