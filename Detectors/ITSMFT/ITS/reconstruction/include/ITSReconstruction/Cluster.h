// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.h
/// \brief Definition of the ITS cluster
#ifndef ALICEO2_ITS_CLUSTER_H
#define ALICEO2_ITS_CLUSTER_H

#include "ITSMFTReconstruction/Cluster.h"
#include "ITSBase/GeometryTGeo.h"

class TGeoHMatrix;

namespace o2
{
namespace ITS
{
/// \class Cluster
/// \brief Cluster class for the ITS
///

class Cluster : public o2::ITSMFT::Cluster
{
 public:
  
  Cluster();
  Cluster(const Cluster& cluster);
  ~Cluster() override;

  Cluster& operator=(const Cluster& cluster) = delete;

  Int_t getLayer() const { return sGeom->getLayer(mVolumeId); }

  void goToFrameGlo();
  void goToFrameLoc();
  void goToFrameTrk();
  void getLocalXYZ(Float_t xyz[3]) const;
  void getTrackingXYZ(Float_t xyz[3]) const;
  //
  virtual void print(Option_t* option = "") const;
  virtual const TGeoHMatrix* getTracking2LocalMatrix() const;
  virtual TGeoHMatrix* getMatrix(Bool_t original = kFALSE) const;
  virtual Bool_t getGlobalXYZ(Float_t xyz[3]) const;
  virtual Bool_t getGlobalCov(Float_t cov[6]) const;
  virtual Bool_t getXRefPlane(Float_t& xref) const;
  virtual Bool_t getXAlphaRefPlane(Float_t& x, Float_t& alpha) const;
  //
  static void setGeom(GeometryTGeo* gm) { sGeom = gm; }
  static void setSortMode(SortMode_t md)
  {
    sMode &= ~kSortBits;
    sMode |= md;
  }
  static UInt_t getSortMode() { return sMode & kSortBits; }
  static UInt_t getMode() { return sMode; }
  static SortMode_t sortModeIdTrkYZ() { return kSortIdTrkYZ; }
  static SortMode_t sortModeIdLocXZ() { return kSortIdLocXZ; }
  //
  virtual Bool_t isSortable() const { return kTRUE; }
  virtual Bool_t isEqual(const TObject* obj) const;
  Int_t Compare(const TObject* obj) const override;
  //
 protected:
  //
  static UInt_t sMode;        //!< general mode (sorting mode etc)
  static GeometryTGeo* sGeom; //!< pointer on the geometry data

  ClassDefOverride(Cluster, 1)
};
}
}

#endif /* ALICEO2_ITS_CLUSTER_H */
