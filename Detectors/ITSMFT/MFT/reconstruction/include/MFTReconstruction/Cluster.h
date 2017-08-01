// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.h
/// \brief Definition of the MFT cluster (ITS)
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#ifndef ALICEO2_MFT_CLUSTER_H
#define ALICEO2_MFT_CLUSTER_H

#include "ITSMFTReconstruction/Cluster.h"
#include "MFTBase/GeometryTGeo.h"

namespace o2 {
namespace MFT {

class Cluster : public o2::ITSMFT::Cluster
{

 public:
  
  Cluster();
  Cluster(const Cluster& cluster);
  ~Cluster() override;
  
  Cluster& operator=(const Cluster& cluster) = delete;

  void transformITStoMFT(const TGeoHMatrix* matSensor, const TGeoHMatrix* matSensorToITS);

  Float_t getMFTLocalX() const { return mMFTLocalX; }
  Float_t getMFTLocalY() const { return mMFTLocalY; }
  Float_t getMFTLocalZ() const { return mMFTLocalZ; }
  Float_t getGlobalX()   const { return mGlobalX; }
  Float_t getGlobalY()   const { return mGlobalY; }
  Float_t getGlobalZ()   const { return mGlobalZ; }

 private:

  Float_t mMFTLocalX;    ///< X of the cluster in the MFT c.s.
  Float_t mMFTLocalY;    ///< Y of the cluster in the MFT c.s.
  Float_t mMFTLocalZ;    ///< Z of the cluster in the MFT c.s.
  Float_t mGlobalX;      ///< X of the cluster in the ALICE c.s.
  Float_t mGlobalY;      ///< Y of the cluster in the ALICE c.s.
  Float_t mGlobalZ;      ///< Z of the cluster in the ALICE c.s.
  
  ClassDefOverride(Cluster,1);

};

}
}

#endif
