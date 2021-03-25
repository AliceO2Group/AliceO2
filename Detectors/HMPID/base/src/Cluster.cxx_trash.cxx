// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDBase/Cluster.h"
#include "HMPIDBase/Param.h"
#include "TRandom.h"
#include "TMath.h"

using namespace o2::hmpid;

ClassImp(o2::hmpid::Cluster);

Cluster::Cluster(Int_t chamber, Int_t size, Int_t NlocMax, Double_t QRaw, Double_t Q, Double_t X, Double_t Y)
  : mChamber(chamber), mSize(size), mNlocMax(NlocMax), mQRaw(QRaw), mQ(Q), mX(X), mY(Y)
{
}
