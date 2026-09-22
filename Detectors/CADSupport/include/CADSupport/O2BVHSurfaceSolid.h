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
/// \since 2026-07

#ifndef ALICEO2_CADSUPPORT_O2BVHSURFACESOLID_
#define ALICEO2_CADSUPPORT_O2BVHSURFACESOLID_

#include "TGeoBBox.h"

#include <array>
#include <iosfwd>
#include <vector>

class TBuffer3D;

namespace o2
{
namespace cad
{

/// One boundary curve of a BVHSurfaceRecord in the flat form ROOT streams: a segment, an arc or a B-spline.
struct BVHSurfaceCurveRecord {
  int kind = 0; ///< PlanarBoundaryCurve::Kind: 0 = Line, 1 = Arc, 2 = BSpline
  double lineStart[2] = {0., 0.};
  double lineEnd[2] = {0., 0.};
  double center[2] = {0., 0.};
  double radius = 0.;
  double startAngle = 0.;
  double endAngle = 0.;
  int degree = 0;              ///< B-spline degree
  std::vector<double> poles;   ///< B-spline control points, flattened (u, v) pairs
  std::vector<double> weights; ///< B-spline weights (empty => non-rational)
  std::vector<double> knots;   ///< B-spline clamped flat knot vector
};

/// The persistent record of one successful Add*Surface call; reading a solid back replays the records.
struct BVHSurfaceRecord {
  enum Kind { PlanarPolygon = 0,
              CurvedPlanar = 1,
              Cylindrical = 2,
              Spherical = 3,
              Conical = 4,
              Toroidal = 5 };

  int kind = PlanarPolygon;
  double origin[3] = {0., 0., 0.}; ///< origin / centerPoint / center
  double axisA[3] = {0., 0., 0.};  ///< axisU / axis / polarAxis
  double axisB[3] = {0., 0., 0.};  ///< axisV / referenceAxisU

  /// The remaining scalar arguments in Add*Surface declaration order (see expectedScalarCount);
  /// the count is checked against the kind on replay.
  std::vector<double> scalars;

  bool innerWall = false;
  bool trimmed = false; ///< the wire-trim overload was used (quadrics only)

  /// The wires, outer first: PlanarPolygon stores (u, v) pairs in polygonPoints, the others curves; wireSizes counts per wire.
  std::vector<double> polygonPoints;
  std::vector<BVHSurfaceCurveRecord> curves;
  std::vector<int> wireSizes;

  /// Sidecar v3 boundary edge identities in curve order: an edge-table index and a BoundaryEdgeFlag byte; empty when not stated.
  std::vector<unsigned int> boundaryEdgeIds;
  std::vector<unsigned char> boundaryEdgeFlags;

  /// How many entries \a scalars must hold for \a kind, or -1 for an unknown kind.
  static int expectedScalarCount(int recordKind);
};

class O2BVHSurfaceSolid : public TGeoBBox
{
 public:
  using Point2D = std::array<double, 2>;
  using Point3D = std::array<double, 3>;

  O2BVHSurfaceSolid();
  explicit O2BVHSurfaceSolid(const char* name);
  ~O2BVHSurfaceSolid() override;

  O2BVHSurfaceSolid(const O2BVHSurfaceSolid&) = delete;
  O2BVHSurfaceSolid& operator=(const O2BVHSurfaceSolid&) = delete;

  bool AddPlanarSurface(const Point3D& origin, const Point3D& axisU, const Point3D& axisV,
                        const std::vector<Point2D>& outerWire,
                        const std::vector<std::vector<Point2D>>& innerWires = {});

  /// One boundary curve in the surface's local (u, v) frame: a line segment, a circular arc or a clamped (rational) B-spline.
  struct PlanarBoundaryCurve {
    enum Kind { Line,
                Arc,
                BSpline };
    Kind kind = Line;
    Point2D lineStart{0., 0.};
    Point2D lineEnd{0., 0.};
    Point2D center{0., 0.};
    double radius = 0.;
    double startAngle = 0.;
    double endAngle = 0.;
    int degree = 0;              ///< B-spline degree
    std::vector<Point2D> poles;  ///< B-spline control points
    std::vector<double> weights; ///< B-spline weights (empty ⇒ non-rational)
    std::vector<double> knots;   ///< B-spline clamped flat knot vector

    static PlanarBoundaryCurve makeLine(const Point2D& start, const Point2D& end)
    {
      PlanarBoundaryCurve curve;
      curve.kind = Line;
      curve.lineStart = start;
      curve.lineEnd = end;
      return curve;
    }
    static PlanarBoundaryCurve makeArc(const Point2D& c, double r, double start, double end)
    {
      PlanarBoundaryCurve curve;
      curve.kind = Arc;
      curve.center = c;
      curve.radius = r;
      curve.startAngle = start;
      curve.endAngle = end;
      return curve;
    }
    static PlanarBoundaryCurve makeBSpline(int splineDegree, std::vector<Point2D> splinePoles,
                                           std::vector<double> splineWeights, std::vector<double> splineKnots)
    {
      PlanarBoundaryCurve curve;
      curve.kind = BSpline;
      curve.degree = splineDegree;
      curve.poles = std::move(splinePoles);
      curve.weights = std::move(splineWeights);
      curve.knots = std::move(splineKnots);
      return curve;
    }
  };

  /// Add an exact planar surface bounded by line/arc wires; axisU and axisV are orthonormal and axisU x axisV points out.
  bool AddCurvedPlanarSurface(const Point3D& origin, const Point3D& axisU, const Point3D& axisV,
                              const std::vector<PlanarBoundaryCurve>& outerWire,
                              const std::vector<std::vector<PlanarBoundaryCurve>>& innerWires = {});

  /// Add a cylindrical wall of \a radius around \a axis over a height range and a phi sweep; innerWall points the normal to the axis.
  bool AddCylindricalSurface(const Point3D& centerPoint, const Point3D& axis, const Point3D& referenceAxisU,
                             double radius, double heightMin, double heightMax, double phiStart = 0.,
                             double phiSweep = 6.283185307179586, bool innerWall = false);

  /// As AddCylindricalSurface, trimmed by line/arc wires in the (phi[rad], h[cm]) domain, which decide containment.
  bool AddCylindricalSurface(const Point3D& centerPoint, const Point3D& axis, const Point3D& referenceAxisU,
                             double radius, double heightMin, double heightMax, double phiStart, double phiSweep,
                             bool innerWall, const std::vector<PlanarBoundaryCurve>& outerTrim,
                             const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims = {});

  /// Add a spherical surface of \a radius trimmed to a theta range and a phi sweep; the defaults give a full sphere.
  bool AddSphericalSurface(const Point3D& center, const Point3D& polarAxis, const Point3D& referenceAxisU,
                           double radius, double thetaMin = 0., double thetaMax = 3.141592653589793,
                           double phiStart = 0., double phiSweep = 6.283185307179586, bool innerWall = false);

  /// As AddSphericalSurface, trimmed by line/arc wires in the (phi[rad], theta[rad]) domain.
  bool AddSphericalSurface(const Point3D& center, const Point3D& polarAxis, const Point3D& referenceAxisU,
                           double radius, double thetaMin, double thetaMax, double phiStart, double phiSweep,
                           bool innerWall, const std::vector<PlanarBoundaryCurve>& outerTrim,
                           const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims = {});

  /// Add a conical wall whose radius runs linearly from \a radiusAtMin to \a radiusAtMax; one radius may be zero.
  bool AddConicalSurface(const Point3D& centerPoint, const Point3D& axis, const Point3D& referenceAxisU,
                         double radiusAtMin, double radiusAtMax, double heightMin, double heightMax,
                         double phiStart = 0., double phiSweep = 6.283185307179586, bool innerWall = false);

  /// As AddConicalSurface, trimmed by line/arc wires in the (phi[rad], h[cm]) domain, which decide containment.
  bool AddConicalSurface(const Point3D& centerPoint, const Point3D& axis, const Point3D& referenceAxisU,
                         double radiusAtMin, double radiusAtMax, double heightMin, double heightMax, double phiStart,
                         double phiSweep, bool innerWall, const std::vector<PlanarBoundaryCurve>& outerTrim,
                         const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims = {});

  /// Add a toroidal surface trimmed to a phiRing x phiTube rectangle; the defaults give a full torus, innerWall points the normal to the tube spine.
  bool AddToroidalSurface(const Point3D& centerPoint, const Point3D& axis, const Point3D& referenceAxisU,
                          double majorRadius, double minorRadius, double phiStart = 0.,
                          double phiSweep = 6.283185307179586, double tubeStart = 0.,
                          double tubeSweep = 6.283185307179586, bool innerWall = false);

  /// As AddToroidalSurface, trimmed by wires in the (phiRing, phiTube) domain; the trim may not wrap more than a turn in either angle.
  bool AddToroidalSurface(const Point3D& centerPoint, const Point3D& axis, const Point3D& referenceAxisU,
                          double majorRadius, double minorRadius, double phiStart, double phiSweep, double tubeStart,
                          double tubeSweep, bool innerWall, const std::vector<PlanarBoundaryCurve>& outerTrim,
                          const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims = {});

  /// \name Boundary edge identity (sidecar v3): when every face states its edges, CloseShape decides closure by counting them
  /// @{
  enum BoundaryEdgeFlag : unsigned char {
    kEdgeReversed = 1u << 0,   ///< the face runs against the edge's own direction
    kEdgeDegenerate = 1u << 1, ///< cone apex / sphere pole: a point, so it has no second face
    kEdgeAnchored = 1u << 2    ///< entry i is trim curve i of this face, so it can be measured
  };

  /// Attach surface \a surfaceIndex's edge identities in trim-curve order; false on a bad index or mismatched lengths.
  bool SetSurfaceBoundaryEdges(int surfaceIndex, const std::vector<unsigned int>& edgeIds,
                               const std::vector<unsigned char>& edgeFlags);
  /// @}

  /// Finalize the shape: bounding box, display mesh, BVH and closure diagnostics, reported when \a check is set.
  void CloseShape(bool check = true);

  int GetNsurfaces() const;
  bool IsDefined() const;

  /// \name The source model's own tolerance, in cm, from the sidecar; zero means not stated
  /// @{
  void SetModelTolerance(double toleranceCm);
  double GetModelTolerance() const { return fModelTolerance; }
  /// @}

  /// Whether the BVH acceleration structure has been built (after CloseShape).
  bool HasBVH() const;
  /// Fill the BVH root-node bounding box; returns false when no BVH has been built.
  bool GetBVHRootBounds(Point3D& lower, Point3D& upper) const;
  /// Test hook: distinct surfaces whose cover boxes the ray traverses; -1 without a BVH.
  int CountBVHRayCandidates(const Point3D& point, const Point3D& direction) const;

  /// Ray tmax tightening in the distance queries, on by default; it never changes an answer. Process-wide, not thread safe.
  static void SetRayTMaxPruning(bool enable);
  static bool GetRayTMaxPruning();

  /// Per-thread count of surfaces handed to the BVH leaf callback by DistFrom* since the last reset.
  static void ResetRayCandidateCounter();
  static long long GetRayCandidateCount();

  /// Per-thread count of surfaces handed to distanceSqToPatch by Safety and ComputeNormal since the last reset.
  static void ResetSafetyCandidateCounter();
  static long long GetSafetyCandidateCount();

  /// Test-only sabotage: prune on the distance to the box centre, which bounds nothing, so the twins must disagree.
  static void SetSafetyBoundUnsoundForTest(bool enable);
  static bool GetSafetyBoundUnsoundForTest();

  /// One crossing of the containment parity ray, as seen by Contains().
  struct ContainsCrossing {
    double distance = 0.;        ///< ray parameter of the hit
    double normalAlignment = 0.; ///< dot(hit normal, test direction): < 0 enters, > 0 exits
    /// The hit lay in its patch's on-boundary band, so a tie-break kept it; Contains() re-shoots on these.
    bool onTrimBoundary = false;
  };

  /// Diagnostic: the parity ray's crossings at \a point from the BVH and from the loop, sorted by distance.
  void DescribeContainsCrossings(const Point3D& point, std::vector<ContainsCrossing>& bvhCrossings,
                                 std::vector<ContainsCrossing>& loopCrossings) const;

  /// As above for an explicit shooting \a direction: the crossing list behind ContainsAlongDirection().
  void DescribeContainsCrossings(const Point3D& point, const Point3D& direction,
                                 std::vector<ContainsCrossing>& bvhCrossings,
                                 std::vector<ContainsCrossing>& loopCrossings) const;

  /// Whether the closed shape forms a closed 2-manifold (every boundary edge shared by two faces).
  /// Meaningful only after CloseShape(); detects e.g. missing faces.
  bool IsClosed() const;
  /// Whether all shared boundary edges are traversed in opposite directions after CloseShape();
  /// detects e.g. reversed faces (inconsistent outward normals).
  bool IsOrientationConsistent() const;

  /// How far navigation can be trusted: parity containment is defined only on a closed, consistently oriented 2-manifold.
  /// Ordered by severity; CloseShape reports the worst defect.
  enum class NavigationReliability {
    Undetermined = 0, ///< CloseShape() has not run yet: no diagnostics exist
    Reliable,         ///< closed, consistently oriented 2-manifold: parity is well defined
    ReversedFaces,    ///< closed, but some rim's partner traverses the shared curve the same
                      ///< way: at least one face's outward normal points inward
    OpenSurfaceSet,   ///< some rim has no other face within the match band (missing faces /
                      ///< trim gaps): parity is undefined in the shadow of every gap along
                      ///< the parity test direction. GetRimReports() names the loops
    NonManifold       ///< some rim has two or more other faces within tolerance (coincident
                      ///< or duplicated faces): parity depends on the order hits are
                      ///< clustered in
  };

  /// The reliability state derived from the last CloseShape(); Undetermined before it has run.
  NavigationReliability GetNavigationReliability() const;
  /// Shorthand for GetNavigationReliability() == NavigationReliability::Reliable. False means the
  /// navigation answers of this solid are not to be trusted anywhere, not just near the defect.
  bool IsNavigable() const;
  /// Short stable identifier of a reliability state ("reliable", "open-surface-set", ...), for
  /// logs and machine-readable reports.
  static const char* GetNavigationReliabilityName(NavigationReliability reliability);

  /// Per-chord closure counts: diagnostics only; GetNavigationReliability() reads the rim counts below.
  int GetBoundaryEdgeCount() const;
  int GetNonManifoldEdgeCount() const;
  int GetReversedEdgeCount() const;

  /// \name The rim-based closure measurement, in cm and per rim; GetNavigationReliability() decides on it
  /// @{
  /// Largest distance from any face's trim boundary to the nearest trim boundary of another face, in cm.
  double GetMaxRimIsolation() const;
  /// \name Closure by edge identity (sidecar v3); when available these decide closure and reliability
  /// @{
  /// Whether the edge identities were complete enough to decide closure by counting.
  bool HasEdgeIdentity() const;
  /// Distinct source edges and their incidence: shared, boundary, non-manifold, reversed and degenerate.
  int GetSourceEdgeCount() const;
  int GetSharedSourceEdgeCount() const;
  int GetBoundarySourceEdgeCount() const;
  int GetNonManifoldSourceEdgeCount() const;
  int GetReversedSourceEdgeCount() const;
  int GetDegenerateSourceEdgeCount() const;
  /// Largest Hausdorff distance between the two faces' realisations of one shared edge, in cm; a measurement only.
  double GetMaxSharedEdgeDeviation() const;
  /// How many shared edges that maximum is over, and how many could not contribute because one of
  /// the two faces carries a parametric-rectangle trim with no per-edge curve to sample.
  int GetMeasuredSharedEdgeCount() const;
  int GetUnmeasuredSharedEdgeCount() const;
  /// @}
  /// Largest distance a rim polyline sits from the smooth rim it samples, in cm.
  double GetRimChordResolution() const;
  /// The declared rim match tolerance in cm, the model's own or a fallback: the floor of each chord's match band.
  double GetRimMatchTolerance() const;
  /// Summed trim-boundary length, and the part with no other face within the match band, in cm.
  double GetTotalRimLength() const;
  double GetUnmatchedRimLength() const;
  /// Rim counts: total, and split by the same four states as the edge counters above.
  int GetRimCount() const;
  int GetMatchedRimCount() const;
  int GetBoundaryRimCount() const;
  int GetNonManifoldRimCount() const;
  int GetReversedRimCount() const;

  /// One trim loop of one face as the closure measurement saw it, naming the rim and its worst chord.
  struct RimReport {
    int surface = -1;      ///< index into GetSurfaceRecords() of the face owning this rim
    int rimOnSurface = -1; ///< which trim loop of that face, in the order the face emits them
    bool closed = false;   ///< the rim polyline returns to its own first point
    int chords = 0;
    int unmatchedChords = 0;     ///< of them, how many found no other face within the tolerance
    double length = 0.;          ///< the rim's length in cm
    double unmatchedLength = 0.; ///< how much of it has no other face within the tolerance, in cm
    /// Largest distance from a chord midpoint of this rim to another face's chord, where, and which face (-1 if none).
    double maxIsolation = 0.;
    std::array<double, 3> maxIsolationPoint{{0., 0., 0.}};
    int maxIsolationFace = -1;
    /// What this rim alone implies about the solid, on the same scale GetNavigationReliability()
    /// reports: Reliable means matched. That call returns exactly the worst state present here.
    NavigationReliability state = NavigationReliability::Undetermined;
  };
  /// Every rim of the last CloseShape(), in the order the faces were visited; empty before it has
  /// run. GetRimCount() is its size.
  const std::vector<RimReport>& GetRimReports() const;

  /// Each face's divergence-theorem contribution to Capacity(), in record order.
  void GetSurfaceCapacityContributions(std::vector<double>& contributions) const;
  /// @}

  void ComputeBBox() override;

  int DistancetoPrimitive(int, int) override { return 99999; }
  const TBuffer3D& GetBuffer3D(int reqSections, Bool_t localFrame) const override;
  void GetMeshNumbers(int& nvert, int& nsegs, int& npols) const override;
  int GetNmeshVertices() const override;

  /// Fill \a array with \a npoints points on the solid's exact boundary; kFALSE below GetNmeshVertices() so ROOT uses SetPoints().
  Bool_t GetPointsOnSegments(Int_t npoints, Double_t* array) const override;

  /// The tolerance GetPointsOnSegments() holds its points to, in cm. A point further than this
  /// from its own patch is replaced by an exact display-mesh vertex rather than emitted.
  static constexpr double kSurfacePointTolerance = 1.e-11;

  void InspectShape() const override {}
  TBuffer3D* MakeBuffer3D() const override;
  void Print(Option_t* option = "") const override;
  void SavePrimitive(std::ostream&, Option_t*) override {}
  void SetPoints(double* points) const override;
  void SetPoints(Float_t* points) const override;
  void SetSegsAndPols(TBuffer3D& buff) const override;
  void Sizeof3D() const override {}

  Double_t DistFromOutside(const Double_t* point, const Double_t* dir, Int_t iact = 1,
                           Double_t step = TGeoShape::Big(), Double_t* safe = nullptr) const override;
  Double_t DistFromInside(const Double_t* point, const Double_t* dir, Int_t iact = 1,
                          Double_t step = TGeoShape::Big(), Double_t* safe = nullptr) const override;
  bool Contains(const Double_t* point) const override;
  /// Trivial non-BVH Contains looping over all surfaces; kept for debugging and
  /// cross-validation of the BVH-accelerated path (see O2Tessellated::Contains_Loop).
  bool Contains_Loop(const Double_t* point) const;
  /// Diagnostic: the parity answer for one explicit \a direction, bypassing Contains()'s re-shoot policy.
  bool ContainsAlongDirection(const Double_t* point, const Double_t* direction) const;
  /// Non-BVH DistFrom* over all surfaces: the oracles the BVH paths must match exactly.
  Double_t DistFromOutside_Loop(const Double_t* point, const Double_t* dir,
                                Double_t stepmax = TGeoShape::Big()) const;
  Double_t DistFromInside_Loop(const Double_t* point, const Double_t* dir,
                               Double_t stepmax = TGeoShape::Big()) const;
  Double_t Safety(const Double_t* point, Bool_t in = kTRUE) const override;
  void ComputeNormal(const Double_t* point, const Double_t* dir, Double_t* norm) const override;
  /// Non-BVH Safety/ComputeNormal over all surfaces: the oracles the BVH traversal must match bit for bit.
  Double_t Safety_Loop(const Double_t* point, Bool_t in = kTRUE) const;
  void ComputeNormal_Loop(const Double_t* point, const Double_t* dir, Double_t* norm) const;
  Double_t Capacity() const override;

  /// The Add*Surface calls this solid was built from, in order.
  const std::vector<BVHSurfaceRecord>& GetSurfaceRecords() const { return fRecords; }

 private:
  /// Containment shared by Contains() and Contains_Loop(): one parity shot if Reliable, else a vote; \a useBVH picks the path.
  bool containsByParity(const Double_t* point, bool useBVH) const;

  /// The normal shared by ComputeNormal() and ComputeNormal_Loop(); \a useLoop picks the all-surfaces scan.
  void computeNormalFrom(const Double_t* point, const Double_t* dir, Double_t* norm, bool useLoop) const;

  /// Replay fRecords through Add*Surface and CloseShape(); false, leaving the solid undefined, when a record fails.
  bool RebuildFromRecords();

  /// Walk \a point onto patch \a surfaceIndex along its normal to kSurfacePointTolerance; false if it does not get there.
  bool ProjectOntoPatch(int surfaceIndex, double* point) const;

  struct Impl;
  Impl* fImpl = nullptr; //! private bounded-surface implementation

  /// The persistent state: everything else is rebuilt from it. See BVHSurfaceRecord.
  std::vector<BVHSurfaceRecord> fRecords;

  /// The source model's declared tolerance in cm; 0 when unknown. See SetModelTolerance.
  double fModelTolerance = 0.;

  ClassDefOverride(O2BVHSurfaceSolid, 3) // BVH surface-bounded shape class
};

} // namespace cad
} // namespace o2

#endif