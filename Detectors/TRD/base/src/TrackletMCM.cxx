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
//  MCM tracklet                                                          //
//                                                                        //
//  Author: J. Klein (Jochen.Klein@cern.ch)                               //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TrackletMCM.h"
#include <fairlogger/Logger.h>
#include "TRDBase/TRDGeometry.h"

using namespace o2::trd;

TrackletMCM::TrackletMCM(unsigned int trackletWord) : mTrackletWord(trackletWord)
{
  // constructor

  mGeo = new TRDGeometry();
  mLabel[0] = -1;
  mLabel[1] = -1;
  mLabel[2] = -1;
}

TrackletMCM::TrackletMCM(unsigned int trackletWord, int hcid) : mHCId(hcid), mTrackletWord(trackletWord)
{
  // constructor

  mGeo = new TRDGeometry();
  mLabel[0] = -1;
  mLabel[1] = -1;
  mLabel[2] = -1;
}

TrackletMCM::TrackletMCM(unsigned int trackletWord, int hcid, int rob, int mcm) : mHCId(hcid), mTrackletWord(trackletWord), mMCM(mcm), mROB(rob)
{
  // constructor

  mGeo = new TRDGeometry();
  mLabel[0] = -1;
  mLabel[1] = -1;
  mLabel[2] = -1;
}

TrackletMCM::TrackletMCM(const TrackletMCM& rhs) : TrackletBase(rhs), mGeo(0x0), mHCId(rhs.mHCId), mTrackletWord(rhs.mTrackletWord), mMCM(rhs.mMCM), mROB(rhs.mROB), mQ0(rhs.mQ0), mQ1(rhs.mQ1), mNHits(rhs.mNHits), mNHits0(rhs.mNHits0), mNHits1(rhs.mNHits1), mSlope(rhs.mSlope), mOffset(rhs.mOffset), mError(rhs.mError), mNClusters(rhs.mNClusters)
{
  // copy constructor

  mGeo = new TRDGeometry();
  mResiduals = rhs.mResiduals;
  mClsCharges = rhs.mClsCharges;
  mLabel = rhs.mLabel;
}

TrackletMCM::~TrackletMCM()
{
  // destructor
  delete mGeo;
}

int TrackletMCM::getYbin() const
{
  // returns (signed) value of Y
  if (mTrackletWord & 0x1000) {
    return -((~(mTrackletWord - 1)) & 0x1fff);
  } else {
    return (mTrackletWord & 0x1fff);
  }
}

int TrackletMCM::getdY() const
{
  // returns (signed) value of the deflection length
  if (mTrackletWord & (1 << 19)) {
    return -((~((mTrackletWord >> 13) - 1)) & 0x7f);
  } else {
    return ((mTrackletWord >> 13) & 0x7f);
  }
}

void TrackletMCM::setLabel(std::array<int, 3>& label)
{
  // set the labels (up to 3)

  mLabel = label;
}

void TrackletMCM::setClusters(std::vector<float>& res, std::vector<float>& q, int n)
{
  mNClusters = n;

  mResiduals = res;
  mClsCharges = q;
}

float TrackletMCM::getX() const
{
  return mGeo->getTime0((mHCId % 12) / 2);
}

float TrackletMCM::getY() const
{
  return (getYbin() * 160e-4);
}
float TrackletMCM::getZ() const
{
  return mGeo->getPadPlane((mHCId % 12) / 2, (mHCId / 12) % 5)->getRowPos(4 * (mROB / 2) + mMCM / 4) -
         mGeo->getPadPlane((mHCId % 12) / 2, (mHCId / 12) % 5)->getRowSize(4 * (mROB / 2) + mMCM / 4) * .5;
}
float TrackletMCM::getLocalZ() const
{
  return getZ() - (mGeo->getPadPlane((mHCId % 12) / 2, (mHCId / 12) % 5)->getRow0() + mGeo->getPadPlane((mHCId % 12) / 2, (mHCId / 12) % 5)->getRowEnd()) / 2.;
}
