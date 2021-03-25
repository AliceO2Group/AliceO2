// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_CLUSTER_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_CLUSTER_H_

#include "CommonDataFormat/TimeStamp.h"
//#include "HMPIDBase/Hit.h"   // for hit
#include "HMPIDBase/Param.h" // for param
#include "TMath.h"

namespace o2
{
namespace hmpid
{
namespace raw
{
/// \class Claster
/// \brief HMPID cluster implementation
class Cluster
{
 public:
  Cluster() = default;

  Cluster(Int_t chamber, Int_t size, Int_t NlocMax, Double_t QRaw, Double_t Q, Double_t X, Double_t Y);
  ~Cluster() = default;

  Int_t getCh() const { return mChamber; }
  void setCh(Int_t chamber) { mChamber = chamber; }

  Int_t getSize() const { return mSize; }
  void setSize(Int_t size) { mSize = size; }

  Int_t getQRaw() const { return mQRaw; }
  void setQRaw(Int_t QRaw) { mQRaw = QRaw; }

  Int_t getQ() const { return mQ; }
  void setQ(Int_t Q) { mQ = Q; }

  Int_t getX() const { return mX; }
  void setX(Int_t X) { mX = X; }

  Int_t getY() const { return mY; }
  void setY(Int_t Y) { mY = Y; }

 protected:
  Int_t mChamber; /// chamber number
  Int_t mSize;    /// size of the formed cluster from which this cluster deduced
  Int_t mNlocMax; /// number of local maxima in formed cluster
  Double_t mQRaw; /// QDC value of the raw cluster
  Double_t mQ;    /// QDC value of the actual cluster
  Double_t mX;    /// local x postion, [cm]
  Double_t mY;    /// local y postion, [cm]

  ClassDefNV(Cluster, 1);
};

} // namespace raw
} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_CLUSTER_H_ */
