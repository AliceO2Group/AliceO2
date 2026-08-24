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

/// \file TGeoGeometryUtils.cxx
/// \author Sandro Wenzel (CERN)
/// \brief Collection of utility functions for TGeo

#include <DetectorsBase/TGeoGeometryUtils.h>
#include <TGeoShape.h>
#include <TGeoTessellated.h>
#include <TGeoBBox.h>
#include <TGeoMatrix.h>
#include <TBuffer3D.h>
#include <TString.h>
#include <cmath>
#include <vector>

namespace o2
{
namespace base
{

namespace
{
// some helpers to interpret TGeo TBuffer3D output
// and convert it to surface triangles (reengineered from TGeo code)

std::vector<int> BuildVertexLoop(const TBuffer3D& buf,
                                 const std::vector<int>& segs)
{
  // adjacency list
  std::unordered_map<int, std::vector<int>> adj;

  for (int s : segs) {
    int a = buf.fSegs[3 * s + 1];
    int b = buf.fSegs[3 * s + 2];
    adj[a].push_back(b);
    adj[b].push_back(a);
  }

  // start from any vertex
  int start = adj.begin()->first;
  int prev = -1;
  int curr = start;

  std::vector<int> loop;

  while (true) {
    loop.push_back(curr);

    const auto& nbrs = adj[curr];
    int next = -1;

    for (int n : nbrs) {
      if (n != prev) {
        next = n;
        break;
      }
    }

    if (next == -1 || next == start) {
      break;
    }

    prev = curr;
    curr = next;
  }
  return loop;
}

std::vector<std::vector<int>> ExtractPolygons(const TBuffer3D& buf)
{
  std::vector<std::vector<int>> polys;
  Int_t idx = 0;

  for (Int_t ip = 0; ip < buf.NbPols(); ++ip) {

    idx++; // color
    Int_t nseg = buf.fPols[idx++];

    std::vector<int> segs(nseg);
    for (Int_t i = 0; i < nseg; ++i) {
      segs[i] = buf.fPols[idx++];
    }

    auto verts = BuildVertexLoop(buf, segs);
    if (verts.size() >= 3) {
      polys.push_back(std::move(verts));
    }
  }

  return polys;
}

std::vector<std::array<int, 3>>
  Triangulate(const std::vector<std::vector<int>>& polys)
{
  std::vector<std::array<int, 3>> tris;
  for (const auto& poly : polys) {
    int nv = poly.size();
    if (nv < 3) {
      continue;
    }

    int v0 = poly[0];
    for (int i = 1; i < nv - 1; ++i) {
      tris.push_back({{v0, poly[i], poly[i + 1]}});
    }
  }
  return tris;
}

TGeoTessellated* MakeTessellated(const TBuffer3D& buf)
{
  auto polys = ExtractPolygons(buf);
  auto tris = Triangulate(polys);
  int i = 0;
  auto* tess = new TGeoTessellated("tess");
  const Double_t* p = buf.fPnts;
  for (auto& t : tris) {
    tess->AddFacet(
      TGeoTessellated::Vertex_t{p[3 * t[0]], p[3 * t[0] + 1], p[3 * t[0] + 2]},
      TGeoTessellated::Vertex_t{p[3 * t[1]], p[3 * t[1] + 1], p[3 * t[1] + 2]},
      TGeoTessellated::Vertex_t{p[3 * t[2]], p[3 * t[2] + 1], p[3 * t[2] + 2]});
  }
  tess->CloseShape();
  return tess;
}
} // end anonymous namespace

///< Transform any (primitive) TGeoShape to a TGeoTessellated
TGeoTessellated* TGeoGeometryUtils::TGeoShapeToTGeoTessellated(TGeoShape const* shape)
{
  auto& buf = shape->GetBuffer3D(TBuffer3D::kRawSizes | TBuffer3D::kRaw | TBuffer3D::kCore, false);
  auto tes = MakeTessellated(buf);
  return tes;
}


///< Bounded stand-in for a TGeoHalfSpace
void TGeoGeometryUtils::makeHalfSpaceBox(const char* name, const double p[3], const double n[3], double reach)
{
  // TGeoHalfSpace contains the points x with (p - x) . n >= 0, and normalizes n itself.
  double nn[3] = {n[0], n[1], n[2]};
  const double norm = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
  for (auto& c : nn) {
    c /= norm;
  }

  // an orthonormal triad (u, v, nn); the seed is chosen to stay away from nn
  double a[3] = {1., 0., 0.};
  if (std::abs(nn[0]) > 0.9) {
    a[0] = 0.;
    a[1] = 1.;
  }
  double u[3] = {a[1] * nn[2] - a[2] * nn[1], a[2] * nn[0] - a[0] * nn[2], a[0] * nn[1] - a[1] * nn[0]};
  const double unorm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  for (auto& c : u) {
    c /= unorm;
  }
  const double v[3] = {nn[1] * u[2] - nn[2] * u[1], nn[2] * u[0] - nn[0] * u[2], nn[0] * u[1] - nn[1] * u[0]};

  // rotation taking the local z axis onto nn (TGeoRotation stores the matrix row-wise,
  // so the images of the local axes are its columns)
  const double m[9] = {u[0], v[0], nn[0], u[1], v[1], nn[1], u[2], v[2], nn[2]};
  auto* rot = new TGeoRotation(TString::Format("%s_rot", name));
  rot->SetMatrix(m);

  // centre the cube one half-size behind the plane, so its +z face lies on the plane
  auto* tr = new TGeoCombiTrans(TString::Format("%s_tr", name), p[0] - reach * nn[0], p[1] - reach * nn[1],
                                p[2] - reach * nn[2], rot);
  tr->RegisterYourself();

  new TGeoBBox(name, reach, reach, reach);
}

} // namespace base
} // namespace o2
