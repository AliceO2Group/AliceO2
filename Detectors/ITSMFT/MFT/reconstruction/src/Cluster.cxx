// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.cxx
/// \brief Implementation of the MFT (ITS) Cluster class
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "MFTReconstruction/Cluster.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Cluster)

//_____________________________________________________________________________
Cluster::Cluster() 
 : o2::ITSMFT::Cluster()
{

  // default constructor

}

//_____________________________________________________________________________
Cluster::~Cluster()
{

  // default destructor

}

//_____________________________________________________________________________
Cluster::Cluster(const Cluster& cluster)
  : o2::ITSMFT::Cluster(cluster)
{

  // copy constructor

}

//_____________________________________________________________________________
void Cluster::transformITStoMFT(const TGeoHMatrix* matSensor, const TGeoHMatrix* matSensorToITS)
{

  Double_t posLocITS[3] = { getX(), getY(), getZ() }, posLocMFT[3], posGlobal[3];
  memset(posLocMFT, 0, sizeof(Double_t) * 3);
  memset(posGlobal, 0, sizeof(Double_t) * 3);

  matSensorToITS->LocalToMaster(posLocITS,posGlobal);
  mGlobalX = posGlobal[0];
  mGlobalY = posGlobal[1];
  mGlobalZ = posGlobal[2];

  matSensor->MasterToLocal(posGlobal,posLocMFT);
  mMFTLocalX = posLocMFT[0];
  mMFTLocalY = posLocMFT[1];
  mMFTLocalZ = posLocMFT[2];

}

