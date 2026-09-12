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

/// \file O2SurfaceSolidIO.cxx
/// \brief Readers of the surface and facet sidecars, in sync with the writers in O2_CADtoTGeo.py.

#include "CADSupport/O2SurfaceSolidIO.h"
#include "CADSupport/O2BVHSurfaceSolid.h"
#include "DetectorsBase/O2Tessellated.h"

#include "BoundedSurface.h"

#include <TError.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace o2
{
namespace cad
{

using o2::base::O2Tessellated;

namespace
{

/// Sidecar versions this reader understands: v2 adds a float64 model tolerance (cm) to the header,
/// v3 a uint32 edge-table size and each face's boundary edge identities after its wires.
constexpr uint32_t kSidecarVersionMin = 1;
constexpr uint32_t kSidecarVersionMax = 3;

/// A version-1 sidecar's model tolerance, in cm: the extractor precision, as a fallback.
constexpr double kSidecarV1FallbackTolerance = 1.e-6;

constexpr uint32_t kFlagInnerWall = 1u << 0;

enum SurfaceType : uint32_t {
  kPlane = 1,
  kCylinder = 2,
  kCone = 3,
  kSphere = 4,
  kTorus = 5,
};

enum CurveType : uint32_t {
  kLineSegment = 0,
  kCircularArc = 1,
  kBSpline2D = 2,
};

/// Parse a B-spline edge record [degree, nPoles, poles, weights, knots] into \a curve; false when malformed.
bool parseBSplineEdge(const std::vector<double>& params, O2BVHSurfaceSolid::PlanarBoundaryCurve& curve)
{
  if (params.size() < 2) {
    return false;
  }
  const int degree = static_cast<int>(std::lround(params[0]));
  const int nPoles = static_cast<int>(std::lround(params[1]));
  if (degree < 1 || nPoles < degree + 1) {
    return false;
  }
  const size_t nKnots = static_cast<size_t>(nPoles) + degree + 1;
  const size_t expected = 2 + 2 * static_cast<size_t>(nPoles) + static_cast<size_t>(nPoles) + nKnots;
  if (params.size() < expected) {
    return false;
  }
  std::vector<O2BVHSurfaceSolid::Point2D> poles(nPoles);
  size_t offset = 2;
  for (int i = 0; i < nPoles; ++i) {
    poles[i] = {params[offset], params[offset + 1]};
    offset += 2;
  }
  std::vector<double> weights(nPoles);
  for (int i = 0; i < nPoles; ++i) {
    weights[i] = params[offset++];
  }
  std::vector<double> knots(nKnots);
  for (size_t i = 0; i < nKnots; ++i) {
    knots[i] = params[offset++];
  }
  curve = O2BVHSurfaceSolid::PlanarBoundaryCurve::makeBSpline(degree, std::move(poles), std::move(weights),
                                                              std::move(knots));
  return true;
}

struct SidecarEdge {
  uint32_t curveType = 0;
  std::vector<double> params;
};

struct SidecarWire {
  uint32_t role = 0; // 0 = outer, 1 = inner
  std::vector<SidecarEdge> edges;
};

/// Packed records: always read field by field.
template <typename T>
bool readValue(std::ifstream& in, T& value)
{
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return static_cast<bool>(in);
}

/// A single field, written on its own; the counterpart of readValue.
template <typename T>
void writeValue(std::ofstream& out, const T& value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

/// Bytes left to read, 0 once the stream is bad; every count read from the file is checked against it.
uint64_t bytesRemaining(std::ifstream& in, std::streamoff fileSize)
{
  if (!in) {
    return 0;
  }
  const std::streamoff here = in.tellg();
  return here < 0 || here > fileSize ? 0 : static_cast<uint64_t>(fileSize - here);
}

bool readDoubles(std::ifstream& in, std::vector<double>& values, uint32_t n, std::streamoff fileSize)
{
  if (static_cast<uint64_t>(n) * sizeof(double) > bytesRemaining(in, fileSize)) {
    return false;
  }
  values.resize(n);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(n) * sizeof(double));
  return static_cast<bool>(in);
}

O2BVHSurfaceSolid::Point3D point3(const std::vector<double>& p, size_t offset)
{
  return {p[offset], p[offset + 1], p[offset + 2]};
}

/// Start/end (u, v) endpoints of a sidecar edge; a B-spline edge's parsed curve goes to \a bspline.
bool edgeEndpoints(const SidecarEdge& edge, O2BVHSurfaceSolid::Point2D& start, O2BVHSurfaceSolid::Point2D& end,
                   O2BVHSurfaceSolid::PlanarBoundaryCurve& bspline)
{
  if (edge.curveType == kLineSegment && edge.params.size() >= 4) {
    start = {edge.params[0], edge.params[1]};
    end = {edge.params[2], edge.params[3]};
    return true;
  }
  if (edge.curveType == kCircularArc && edge.params.size() >= 5) {
    const double cu = edge.params[0], cv = edge.params[1], r = edge.params[2];
    const double a0 = edge.params[3], a1 = edge.params[3] + edge.params[4];
    start = {cu + r * std::cos(a0), cv + r * std::sin(a0)};
    end = {cu + r * std::cos(a1), cv + r * std::sin(a1)};
    return true;
  }
  if (edge.curveType == kBSpline2D) {
    if (!parseBSplineEdge(edge.params, bspline)) {
      return false;
    }
    // Evaluate the curve rather than read its first and last poles, which lie off the curve for an
    // unclamped or periodic knot vector.
    std::vector<surface::Vec2> poles;
    poles.reserve(bspline.poles.size());
    for (const auto& pole : bspline.poles) {
      poles.push_back({pole[0], pole[1]});
    }
    const surface::Curve2D evaluated =
      surface::Curve2D::makeBSpline(bspline.degree, std::move(poles), bspline.weights, bspline.knots);
    const surface::Vec2 first = evaluated.startPoint();
    const surface::Vec2 last = evaluated.endPoint();
    start = {first.uCoord, first.vCoord};
    end = {last.uCoord, last.vCoord};
    return true;
  }
  return false;
}

/// The first fundamental form of a sidecar record's surface, from its own parameters, for the join check.
struct RecordMetric {
  uint32_t surfaceType = 0;
  const double* params = nullptr;

  static void evaluate(const void* context, const surface::Vec2& uv, double& gUU, double& gUV, double& gVV)
  {
    const auto& record = *static_cast<const RecordMetric*>(context);
    const double* p = record.params;
    switch (record.surfaceType) {
      case kPlane:
        surface::planeParametricMetric({p[3], p[4], p[5]}, {p[6], p[7], p[8]}, gUU, gUV, gVV);
        return;
      case kCylinder:
        surface::cylinderParametricMetric(p[9], gUU, gUV, gVV);
        return;
      case kCone: {
        // r(h) = radiusAtMin + slope * (h - heightMin), with slope from the two radii/heights
        const double slope = (p[10] - p[9]) / (p[12] - p[11]);
        surface::coneParametricMetric(p[9] + slope * (uv.vCoord - p[11]), slope, gUU, gUV, gVV);
        return;
      }
      case kSphere:
        surface::sphereParametricMetric(p[9], uv.vCoord, gUU, gUV, gVV);
        return;
      case kTorus:
        surface::torusParametricMetric(p[9], p[10], uv.vCoord, gUU, gUV, gVV);
        return;
      default:
        // an unknown type is rejected further down; the identity keeps this total meanwhile
        gUU = 1.;
        gUV = 0.;
        gVV = 1.;
        return;
    }
  }

  surface::ParametricMetric metric() const { return {&evaluate, this}; }
};

/// Convert a sidecar wire into a PlanarBoundaryCurve loop; joins are judged as 3D gaps in cm against the kernel's band.
/// \a anyArc is set by a curved edge; \a toleranceOrigin names the band for the diagnostic.
bool wireToCurves(const std::string& file, size_t surfaceIndex, const SidecarWire& wire,
                  std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>& curves, bool& anyArc,
                  const surface::ParametricMetric& metric, double joinTolerance, const char* toleranceOrigin)
{
  using Curve = O2BVHSurfaceSolid::PlanarBoundaryCurve;
  curves.clear();
  curves.reserve(wire.edges.size());
  // every edge's endpoints, and a B-spline edge's parsed curve, computed once
  const size_t nEdges = wire.edges.size();
  std::vector<O2BVHSurfaceSolid::Point2D> starts(nEdges);
  std::vector<O2BVHSurfaceSolid::Point2D> ends(nEdges);
  std::vector<Curve> bsplines(nEdges);
  for (size_t e = 0; e < nEdges; ++e) {
    if (!edgeEndpoints(wire.edges[e], starts[e], ends[e], bsplines[e])) {
      ::Error("LoadSurfaceSolid", "%s: surface %zu: unsupported or malformed wire edge %zu", file.c_str(),
              surfaceIndex, e);
      return false;
    }
  }
  for (size_t e = 0; e < nEdges; ++e) {
    const auto& edge = wire.edges[e];
    const auto& end = ends[e];
    const auto& nextStart = starts[(e + 1) % nEdges];
    const double joinGapSq = metric.distanceSq({end[0], end[1]}, {nextStart[0], nextStart[1]});
    if (joinGapSq > joinTolerance * joinTolerance) {
      ::Error("LoadSurfaceSolid",
              "%s: surface %zu: wire edge %zu end does not join the next edge start (gap %.3g cm, tolerance %.3g cm, "
              "%s)",
              file.c_str(), surfaceIndex, e, std::sqrt(joinGapSq), joinTolerance, toleranceOrigin);
      return false;
    }
    if (edge.curveType == kCircularArc) {
      anyArc = true;
      curves.push_back(Curve::makeArc({edge.params[0], edge.params[1]}, edge.params[2], edge.params[3],
                                      edge.params[3] + edge.params[4]));
    } else if (edge.curveType == kBSpline2D) {
      anyArc = true; // a bspline is a curved edge, so route the plane through AddCurvedPlanarSurface
      curves.push_back(std::move(bsplines[e]));
    } else {
      curves.push_back(Curve::makeLine(starts[e], end));
    }
  }
  return true;
}

/// The two error texts of a trim block, as printf formats taking the file and the surface index.
struct TrimWording {
  const char* moreThanOneOuter;
  const char* noOuter;
};
constexpr TrimWording kPlaneWording{"%s: plane surface %zu has more than one outer wire",
                                    "%s: plane surface %zu has no outer wire"};
constexpr TrimWording kQuadricWording{"%s: quadric surface %zu has more than one outer trim wire",
                                      "%s: quadric surface %zu trim block has no outer wire"};

/// Collect a wire block into one outer and several inner PlanarBoundaryCurve loops in the (u, v) domain; \a anyArc is set by a curved edge.
bool collectTrim(const std::string& file, size_t surfaceIndex, const TrimWording& wording,
                 const std::vector<SidecarWire>& wires, std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>& outer,
                 std::vector<std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>>& inners, bool& anyArc,
                 const surface::ParametricMetric& metric, double joinTolerance, const char* toleranceOrigin)
{
  bool haveOuter = false;
  for (const auto& wire : wires) {
    std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve> curves;
    if (!wireToCurves(file, surfaceIndex, wire, curves, anyArc, metric, joinTolerance, toleranceOrigin)) {
      return false;
    }
    if (wire.role == 0) {
      if (haveOuter) {
        ::Error("LoadSurfaceSolid", wording.moreThanOneOuter, file.c_str(), surfaceIndex);
        return false;
      }
      outer = std::move(curves);
      haveOuter = true;
    } else {
      inners.push_back(std::move(curves));
    }
  }
  if (!haveOuter) {
    ::Error("LoadSurfaceSolid", wording.noOuter, file.c_str(), surfaceIndex);
    return false;
  }
  return true;
}

/// Permute a face's edge identities from the sidecar's wire order into the kernel's: the outer wire first, then the inner wires.
void reorderEdgeRefsToKernelOrder(const std::vector<SidecarWire>& wires, std::vector<unsigned int>& edgeIds,
                                  std::vector<unsigned char>& edgeFlags)
{
  size_t totalEdges = 0;
  for (const auto& wire : wires) {
    totalEdges += wire.edges.size();
  }
  if (wires.empty() || totalEdges != edgeIds.size()) {
    return;
  }
  // kernel offset of each sidecar wire: the outer wire first, then the inner wires in file order
  std::vector<size_t> kernelOffset(wires.size(), 0);
  size_t running = 0;
  for (size_t w = 0; w < wires.size(); ++w) {
    if (wires[w].role == 0) {
      kernelOffset[w] = 0;
      running = wires[w].edges.size();
      break;
    }
  }
  for (size_t w = 0; w < wires.size(); ++w) {
    if (wires[w].role != 0) {
      kernelOffset[w] = running;
      running += wires[w].edges.size();
    }
  }

  std::vector<unsigned int> permutedIds(edgeIds.size());
  std::vector<unsigned char> permutedFlags(edgeFlags.size());
  size_t sidecarOffset = 0;
  for (size_t w = 0; w < wires.size(); ++w) {
    for (size_t e = 0; e < wires[w].edges.size(); ++e) {
      permutedIds[kernelOffset[w] + e] = edgeIds[sidecarOffset + e];
      permutedFlags[kernelOffset[w] + e] = edgeFlags[sidecarOffset + e];
    }
    sidecarOffset += wires[w].edges.size();
  }
  edgeIds.swap(permutedIds);
  edgeFlags.swap(permutedFlags);
}

} // namespace

bool LoadSurfaceSolid(const std::string& file, O2BVHSurfaceSolid& solid)
{
  std::ifstream in(file, std::ios::binary);
  if (!in) {
    ::Error("LoadSurfaceSolid", "Cannot open surface sidecar file %s", file.c_str());
    return false;
  }

  in.seekg(0, std::ios::end);
  const std::streamoff fileSize = in.tellg();
  in.seekg(0, std::ios::beg);

  char magic[4];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "O2SS", 4) != 0) {
    ::Error("LoadSurfaceSolid", "%s is not a surface sidecar file (bad magic)", file.c_str());
    return false;
  }

  uint32_t version = 0, nSurfaces = 0, reserved = 0;
  if (!readValue(in, version) || !readValue(in, nSurfaces) || !readValue(in, reserved)) {
    ::Error("LoadSurfaceSolid", "%s: truncated header", file.c_str());
    return false;
  }
  if (version < kSidecarVersionMin || version > kSidecarVersionMax) {
    ::Error("LoadSurfaceSolid", "%s: unsupported sidecar version %u (reader supports %u..%u)", file.c_str(), version,
            kSidecarVersionMin, kSidecarVersionMax);
    return false;
  }

  uint32_t nModelEdges = 0;
  if (version >= 2) {
    double modelTolerance = 0.;
    if (!readValue(in, modelTolerance)) {
      ::Error("LoadSurfaceSolid", "%s: truncated version-2 header (no model tolerance)", file.c_str());
      return false;
    }
    solid.SetModelTolerance(modelTolerance);
    if (version >= 3 && !readValue(in, nModelEdges)) {
      ::Error("LoadSurfaceSolid", "%s: truncated version-3 header (no edge table size)", file.c_str());
      return false;
    }
  } else {
    ::Warning("LoadSurfaceSolid",
              "%s is a version-1 sidecar and states no model tolerance; assuming %g cm (the extractor's precision). "
              "Re-run the converter to record the model's own value.",
              file.c_str(), kSidecarV1FallbackTolerance);
    solid.SetModelTolerance(kSidecarV1FallbackTolerance);
  }

  // the wire-join band, from the header: the band the kernel's Add*Surface applies to the same wires
  const double joinTolerance = surface::wireJoinToleranceFor(solid.GetModelTolerance());
  const char* toleranceOrigin = joinTolerance > surface::kWireJoinTolerance
                                  ? "declared by the model"
                                  : "the extractor-precision fallback";

  for (size_t s = 0; s < nSurfaces; ++s) {
    uint32_t surfaceType = 0, flags = 0, nParams = 0;
    if (!readValue(in, surfaceType) || !readValue(in, flags) || !readValue(in, nParams)) {
      ::Error("LoadSurfaceSolid", "%s: truncated surface record %zu", file.c_str(), s);
      return false;
    }
    std::vector<double> p;
    if (!readDoubles(in, p, nParams, fileSize)) {
      ::Error("LoadSurfaceSolid", "%s: truncated parameters of surface %zu", file.c_str(), s);
      return false;
    }

    // The wire block is self-describing; read it unconditionally.
    uint32_t nWires = 0;
    if (!readValue(in, nWires)) {
      ::Error("LoadSurfaceSolid", "%s: truncated wire count of surface %zu", file.c_str(), s);
      return false;
    }
    // 8 bytes of header per wire is the floor, so a count beyond that cannot be honest
    if (static_cast<uint64_t>(nWires) * 8u > bytesRemaining(in, fileSize)) {
      ::Error("LoadSurfaceSolid", "%s: surface %zu claims %u wires, more than the file holds", file.c_str(), s, nWires);
      return false;
    }
    std::vector<SidecarWire> wires(nWires);
    for (auto& wire : wires) {
      uint32_t nEdges = 0;
      if (!readValue(in, wire.role) || !readValue(in, nEdges)) {
        ::Error("LoadSurfaceSolid", "%s: truncated wire header in surface %zu", file.c_str(), s);
        return false;
      }
      if (static_cast<uint64_t>(nEdges) * 8u > bytesRemaining(in, fileSize)) {
        ::Error("LoadSurfaceSolid", "%s: surface %zu claims %u wire edges, more than the file holds", file.c_str(), s,
                nEdges);
        return false;
      }
      wire.edges.resize(nEdges);
      for (auto& edge : wire.edges) {
        uint32_t nCurveParams = 0;
        if (!readValue(in, edge.curveType) || !readValue(in, nCurveParams) ||
            !readDoubles(in, edge.params, nCurveParams, fileSize)) {
          ::Error("LoadSurfaceSolid", "%s: truncated edge record in surface %zu", file.c_str(), s);
          return false;
        }
      }
    }

    // Version 3: the face's boundary edge identities, in the sidecar's own wire order.
    std::vector<unsigned int> edgeIds;
    std::vector<unsigned char> edgeFlags;
    if (version >= 3) {
      uint32_t nEdgeRefs = 0;
      if (!readValue(in, nEdgeRefs)) {
        ::Error("LoadSurfaceSolid", "%s: truncated edge identity count of surface %zu", file.c_str(), s);
        return false;
      }
      if (static_cast<uint64_t>(nEdgeRefs) * 5u > bytesRemaining(in, fileSize)) {
        ::Error("LoadSurfaceSolid", "%s: surface %zu claims %u edge identities, more than the file holds",
                file.c_str(), s, nEdgeRefs);
        return false;
      }
      edgeIds.resize(nEdgeRefs);
      edgeFlags.resize(nEdgeRefs);
      for (uint32_t e = 0; e < nEdgeRefs; ++e) {
        uint32_t edgeId = 0;
        uint8_t edgeFlag = 0;
        if (!readValue(in, edgeId) || !readValue(in, edgeFlag)) {
          ::Error("LoadSurfaceSolid", "%s: truncated edge identity %u of surface %zu", file.c_str(), e, s);
          return false;
        }
        if (nModelEdges > 0 && edgeId >= nModelEdges) {
          ::Error("LoadSurfaceSolid", "%s: surface %zu edge identity %u is %u, outside the model's %u edge(s)",
                  file.c_str(), s, e, edgeId, nModelEdges);
          return false;
        }
        edgeIds[e] = edgeId;
        edgeFlags[e] = edgeFlag;
      }
    }

    const bool innerWall = (flags & kFlagInnerWall) != 0;
    const RecordMetric recordMetric{surfaceType, p.data()};
    bool added = false;

    // one quadric: check the parameter count, then add the surface untrimmed or with its trim block
    const auto addQuadric = [&](const char* name, uint32_t expectedParams, const auto& addUntrimmed,
                                const auto& addTrimmed) {
      if (nParams != expectedParams) {
        ::Error("LoadSurfaceSolid", "%s: %s surface %zu has %u parameters, expected %u", file.c_str(), name, s,
                nParams, expectedParams);
        return false;
      }
      if (wires.empty()) {
        added = addUntrimmed();
        return true;
      }
      std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve> outer;
      std::vector<std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>> inners;
      bool anyArc = false; // quadric domains accept both line and arc trim edges
      if (!collectTrim(file, s, kQuadricWording, wires, outer, inners, anyArc, recordMetric.metric(), joinTolerance,
                       toleranceOrigin)) {
        return false;
      }
      added = addTrimmed(outer, inners);
      return true;
    };

    switch (surfaceType) {
      case kPlane: {
        if (nParams != 9) {
          ::Error("LoadSurfaceSolid", "%s: plane surface %zu has %u parameters, expected 9", file.c_str(), s, nParams);
          return false;
        }
        // Read every wire as a general line/arc loop. A pure line-segment loop keeps the
        // polygon path (AddPlanarSurface, general-metric); any arc routes to the curved path.
        std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve> outer;
        std::vector<std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>> inners;
        bool anyArc = false;
        if (!collectTrim(file, s, kPlaneWording, wires, outer, inners, anyArc, recordMetric.metric(), joinTolerance,
                         toleranceOrigin)) {
          return false;
        }
        if (anyArc) {
          added = solid.AddCurvedPlanarSurface(point3(p, 0), point3(p, 3), point3(p, 6), outer, inners);
        } else {
          const auto toPolygon = [](const std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>& curves) {
            std::vector<O2BVHSurfaceSolid::Point2D> polygon;
            polygon.reserve(curves.size());
            for (const auto& c : curves) {
              polygon.push_back(c.lineStart);
            }
            return polygon;
          };
          std::vector<std::vector<O2BVHSurfaceSolid::Point2D>> innerPolys;
          innerPolys.reserve(inners.size());
          for (const auto& inner : inners) {
            innerPolys.push_back(toPolygon(inner));
          }
          added = solid.AddPlanarSurface(point3(p, 0), point3(p, 3), point3(p, 6), toPolygon(outer), innerPolys);
        }
        break;
      }
      case kCylinder:
        if (!addQuadric(
              "cylinder", 14,
              [&] {
                return solid.AddCylindricalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                                   p[13], innerWall);
              },
              [&](const auto& outer, const auto& inners) {
                return solid.AddCylindricalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                                   p[13], innerWall, outer, inners);
              })) {
          return false;
        }
        break;
      case kCone:
        if (!addQuadric(
              "cone", 15,
              [&] {
                return solid.AddConicalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                               p[13], p[14], innerWall);
              },
              [&](const auto& outer, const auto& inners) {
                return solid.AddConicalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                               p[13], p[14], innerWall, outer, inners);
              })) {
          return false;
        }
        break;
      case kSphere:
        if (!addQuadric(
              "sphere", 14,
              [&] {
                return solid.AddSphericalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                                 p[13], innerWall);
              },
              [&](const auto& outer, const auto& inners) {
                return solid.AddSphericalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                                 p[13], innerWall, outer, inners);
              })) {
          return false;
        }
        break;
      case kTorus:
        if (!addQuadric(
              "torus", 15,
              [&] {
                return solid.AddToroidalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                                p[13], p[14], innerWall);
              },
              [&](const auto& outer, const auto& inners) {
                return solid.AddToroidalSurface(point3(p, 0), point3(p, 3), point3(p, 6), p[9], p[10], p[11], p[12],
                                                p[13], p[14], innerWall, outer, inners);
              })) {
          return false;
        }
        break;
      default:
        ::Error("LoadSurfaceSolid", "%s: surface %zu has unknown surface type %u", file.c_str(), s, surfaceType);
        return false;
    }

    if (!added) {
      ::Error("LoadSurfaceSolid", "%s: surface %zu was rejected by O2BVHSurfaceSolid", file.c_str(), s);
      return false;
    }
    if (!edgeIds.empty()) {
      reorderEdgeRefsToKernelOrder(wires, edgeIds, edgeFlags);
      solid.SetSurfaceBoundaryEdges(static_cast<int>(s), edgeIds, edgeFlags);
    }
  }

  return true;
}

bool LoadFacetSolid(const std::string& file, O2Tessellated& solid)
{
  std::ifstream in(file, std::ios::binary);
  if (!in) {
    ::Error("LoadFacetSolid", "Cannot open facet sidecar file %s", file.c_str());
    return false;
  }

  in.seekg(0, std::ios::end);
  const std::streamoff fileSize = in.tellg();
  in.seekg(0, std::ios::beg);

  uint32_t nTriangles = 0;
  if (!readValue(in, nTriangles)) {
    ::Error("LoadFacetSolid", "%s: truncated header", file.c_str());
    return false;
  }

  // one record is nine float32; the count is checked against the file before one block read
  const uint64_t recordsBytes = static_cast<uint64_t>(nTriangles) * 9u * sizeof(float);
  const uint64_t remaining = bytesRemaining(in, fileSize);
  if (recordsBytes > remaining) {
    ::Error("LoadFacetSolid", "%s: truncated: %u facet record(s) need %llu byte(s), found %llu", file.c_str(),
            nTriangles, static_cast<unsigned long long>(recordsBytes), static_cast<unsigned long long>(remaining));
    return false;
  }
  std::vector<float> records(9 * static_cast<size_t>(nTriangles));
  in.read(reinterpret_cast<char*>(records.data()), static_cast<std::streamsize>(recordsBytes));
  if (!in) {
    ::Error("LoadFacetSolid", "%s: truncated facet records", file.c_str());
    return false;
  }

  uint32_t nDegenerate = 0;
  for (uint32_t i = 0; i < nTriangles; ++i) {
    const float* v = &records[9 * static_cast<size_t>(i)];
    const O2Tessellated::Vertex_t p0(v[0], v[1], v[2]);
    const O2Tessellated::Vertex_t p1(v[3], v[4], v[5]);
    const O2Tessellated::Vertex_t p2(v[6], v[7], v[8]);
    if (!solid.AddFacet(p0, p1, p2)) {
      // a degenerate facet is a mesh property, not a format error: count it and carry on
      ++nDegenerate;
      continue;
    }
  }
  if (nDegenerate > 0) {
    ::Warning("LoadFacetSolid", "%s: skipped %u degenerate facet(s) of %u", file.c_str(), nDegenerate, nTriangles);
  }

  return true;
}

} // namespace cad
} // namespace o2
