// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TRD tracklet                                                           //
// class for TRD tracklets from TRAP chip                                 //
//                                                                        //
// Authors                                                                //
//  Alex Bercuci (A.Bercuci@gsi.de)                                       //
//  Jochen Klein (jochen.klein@cern.ch)                                   //
//  S. Murray (murrays@cern.ch)                                           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TRDBase/Tracklet.h"
#include <fairlogger/Logger.h>
#include "TRDBase/TRDGeometry.h"

//_____________________________________________________________________________

using namespace std;
using namespace o2::trd;

Tracklet::Tracklet(unsigned int trackletWord) : mTrackletWord(trackletWord)
{
  // constructor

  mGeo = TRDGeometry::instance();
  mLabel[0] = -1;
  mLabel[1] = -1;
  mLabel[2] = -1;
}

Tracklet::Tracklet(unsigned int trackletWord, int hcid) : mHCId(hcid), mTrackletWord(trackletWord)
{
  // constructor

  mGeo = TRDGeometry::instance();
  mLabel[0] = -1;
  mLabel[1] = -1;
  mLabel[2] = -1;
}

Tracklet::Tracklet(unsigned int trackletWord, int hcid, int rob, int mcm) : mHCId(hcid), mTrackletWord(trackletWord), mMCM(mcm), mROB(rob)
{
  // constructor

  mGeo = TRDGeometry::instance();
  mLabel[0] = -1;
  mLabel[1] = -1;
  mLabel[2] = -1;
}

Tracklet::Tracklet(const Tracklet& rhs) : mHCId(rhs.mHCId), mTrackletWord(rhs.mTrackletWord), mMCM(rhs.mMCM), mROB(rhs.mROB), mQ0(rhs.mQ0), mQ1(rhs.mQ1), mNHits(rhs.mNHits), mNHits0(rhs.mNHits0), mNHits1(rhs.mNHits1), mSlope(rhs.mSlope), mOffset(rhs.mOffset), mError(rhs.mError), mNClusters(rhs.mNClusters)
{
  // copy constructor

  mGeo = TRDGeometry::instance();
  mResiduals = rhs.mResiduals;
  mClsCharges = rhs.mClsCharges;
  mLabel = rhs.mLabel;
}

int Tracklet::getYbin() const
{
  // returns (signed) value of Y
  if (mTrackletWord & 0x1000) {
    return -((~(mTrackletWord - 1)) & 0x1fff);
  } else {
    return (mTrackletWord & 0x1fff);
  }
}

int Tracklet::getdY() const
{
  // returns (signed) value of the deflection length
  if (mTrackletWord & (1 << 19)) {
    return -((~((mTrackletWord >> 13) - 1)) & 0x7f);
  } else {
    return ((mTrackletWord >> 13) & 0x7f);
  }
}

void Tracklet::setLabel(std::array<int, 3>& label)
{
  // set the labels (up to 3)

  mLabel = label;
}

void Tracklet::setClusters(std::vector<float>& res, std::vector<float>& q, int n)
{
  mNClusters = n;

  mResiduals = res;
  mClsCharges = q;
}

float Tracklet::getX() const
{
  return mGeo->getTime0((mHCId % 12) / 2);
}

float Tracklet::getY() const
{
  return (getYbin() * 160e-4);
}
float Tracklet::getZ() const
{
  return mGeo->getRowPos((mHCId % 12) / 2, (mHCId / 12) % 5, 4 * (mROB / 2) + mMCM / 4) -
         mGeo->getRowSize((mHCId % 12) / 2, (mHCId / 12) % 5, 4 * (mROB / 2) + mMCM / 4) * .5;
}
float Tracklet::getLocalZ() const
{
  return getZ() - mGeo->getRow0((mHCId % 12) / 2, (mHCId / 12) % 5) + mGeo->getRowEnd((mHCId % 12) / 2, (mHCId / 12) % 5) / 2.;
}
