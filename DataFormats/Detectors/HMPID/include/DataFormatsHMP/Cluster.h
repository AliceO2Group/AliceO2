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

namespace o2
{
namespace hmpid
{
/// \class Cluster
/// \brief HMPID cluster implementation
class Cluster
{
 public:
  Cluster() = default;

  Cluster(int chamber, int size, int NlocMax, float QRaw, float Q, float X, float Y);
  ~Cluster() = default;

  int getCh() const { return mChamber; }
  void setCh(int chamber) { mChamber = chamber; }

  int getSize() const { return mSize; }
  void setSize(int size) { mSize = size; }

  int getQRaw() const { return mQRaw; }
  void setQRaw(int QRaw) { mQRaw = QRaw; }

  int getQ() const { return mQ; }
  void setQ(int Q) { mQ = Q; }

  int getX() const { return mX; }
  void setX(int X) { mX = X; }

  int getY() const { return mY; }
  void setY(int Y) { mY = Y; }

 protected:
  int mChamber; /// chamber number
  int mSize;    /// size of the formed cluster from which this cluster deduced
  int mNlocMax; /// number of local maxima in formed cluster
  float mQRaw;  /// QDC value of the raw cluster
  float mQ;     /// QDC value of the actual cluster
  float mX;     /// local x postion, [cm]
  float mY;     /// local y postion, [cm]

  ClassDefNV(Cluster, 1);
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_CLUSTER_H_ */
