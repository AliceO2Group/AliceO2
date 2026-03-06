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

#ifndef ALICEO2_BASE_O2TESSELLATED_
#define ALICEO2_BASE_O2TESSELLATED_

#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TGeoVector3.h"
#include "TGeoTypedefs.h"
#include "TGeoTessellated.h"

namespace o2
{
namespace base
{

class O2Tessellated : public TGeoBBox
{

 public:
  using Vertex_t = Tessellated::Vertex_t;

 private:
  int fNfacets = 0;         // Number of facets
  int fNvert = 0;           // Number of vertices
  int fNseg = 0;            // Number of segments
  bool fDefined = false;    //! Shape fully defined
  bool fClosedBody = false; // The faces are making a closed body

  // for now separate vectors but might be better to group per face
  std::vector<Vertex_t> fVertices;       // List of vertices
  std::vector<TGeoFacet> fFacets;        // List of facets
  std::vector<Vertex_t> fOutwardNormals; // Vector of outward-facing normals (to be streamed !)

  std::multimap<long, int> fVerticesMap; //! Temporary map used to deduplicate vertices
  bool fIsClosed = false;                //! to know if shape still needs closure/initialization
  void* fBVH = nullptr;                  //! BVH acceleration structure for safety and navigation

  O2Tessellated(const O2Tessellated&) = delete;
  O2Tessellated& operator=(const O2Tessellated&) = delete;

  // bvh helper functions
  void BuildBVH();
  void CalculateNormals();

 public:
  // constructors
  O2Tessellated() {}
  O2Tessellated(const char* name, int nfacets = 0);
  O2Tessellated(const char* name, const std::vector<Vertex_t>& vertices);
  // from a TGeoTessellated
  O2Tessellated(TGeoTessellated const&, bool check = false);

  // destructor
  ~O2Tessellated() override {}

  void ComputeBBox() override;
  void CloseShape(bool check = true, bool fixFlipped = true, bool verbose = true);

  bool AddFacet(const Vertex_t& pt0, const Vertex_t& pt1, const Vertex_t& pt2);
  bool AddFacet(const Vertex_t& pt0, const Vertex_t& pt1, const Vertex_t& pt2, const Vertex_t& pt3);
  bool AddFacet(int i1, int i2, int i3);
  bool AddFacet(int i1, int i2, int i3, int i4);
  int AddVertex(const Vertex_t& vert);

  bool FacetCheck(int ifacet) const;
  Vertex_t FacetComputeNormal(int ifacet, bool& degenerated) const;

  int GetNfacets() const { return fFacets.size(); }
  int GetNsegments() const { return fNseg; }
  int GetNvertices() const { return fNvert; }
  bool IsClosedBody() const { return fClosedBody; }
  bool IsDefined() const { return fDefined; }

  const TGeoFacet& GetFacet(int i) const { return fFacets[i]; }
  const Vertex_t& GetVertex(int i) const { return fVertices[i]; }

  int DistancetoPrimitive(int, int) override { return 99999; }
  const TBuffer3D& GetBuffer3D(int reqSections, Bool_t localFrame) const override;
  void GetMeshNumbers(int& nvert, int& nsegs, int& npols) const override;
  int GetNmeshVertices() const override { return fNvert; }
  void InspectShape() const override {}
  TBuffer3D* MakeBuffer3D() const override;
  void Print(Option_t* option = "") const override;
  void SavePrimitive(std::ostream&, Option_t*) override {}
  void SetPoints(double* points) const override;
  void SetPoints(Float_t* points) const override;
  void SetSegsAndPols(TBuffer3D& buff) const override;
  void Sizeof3D() const override {}

  /// Resize and center the shape in a box of size maxsize
  void ResizeCenter(double maxsize);

  /// Flip all facets
  void FlipFacets()
  {
    for (auto facet : fFacets)
      facet.Flip();
  }

  bool CheckClosure(bool fixFlipped = true, bool verbose = true);

  /// Reader from .obj format
  static O2Tessellated* ImportFromObjFormat(const char* objfile, bool check = false, bool verbose = false);

  // navigation functions used by TGeoNavigator (attention: only the iact == 3 cases implemented for now)
  Double_t DistFromOutside(const Double_t* point, const Double_t* dir, Int_t iact = 1,
                           Double_t step = TGeoShape::Big(), Double_t* safe = nullptr) const override;
  Double_t DistFromInside(const Double_t* point, const Double_t* dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                          Double_t* safe = nullptr) const override;
  bool Contains(const Double_t* point) const override;
  Double_t Safety(const Double_t* point, Bool_t in = kTRUE) const override;
  void ComputeNormal(const Double_t* point, const Double_t* dir, Double_t* norm) const override;

  // these are trivial implementations, just for debugging
  Double_t DistFromInside_Loop(const Double_t* point, const Double_t* dir) const;
  Double_t DistFromOutside_Loop(const Double_t* point, const Double_t* dir) const;
  bool Contains_Loop(const Double_t* point) const;

  Double_t Capacity() const override;

 private:
  // a safety kernel used in multiple implementations
  template <bool closest_facet = false>
  Double_t SafetyKernel(const Double_t* point, bool in, int* closest_facet_id = nullptr) const;

  ClassDefOverride(O2Tessellated, 1) // tessellated shape class
};

} // namespace base
} // namespace o2

#endif
