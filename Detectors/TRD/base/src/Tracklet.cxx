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
#include "TRDBase/Geometry.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
//_____________________________________________________________________________

using namespace o2::trd;

Tracklet::Tracklet()
{
  // constructor
  mGeo = Geometry::instance();
}

Tracklet::Tracklet(int hcid) : mHCId(hcid)
{
  // constructor
  mGeo = Geometry::instance();
}

Tracklet::Tracklet(int hcid, int rob, int mcm, float pid, float rawslope, float rawoffset, float rawslope4trackletword, float rawoffset4trackletword) : mHCId(hcid), mMCM(mcm), mROB(rob), mPID(pid), mdY(rawslope), mY(rawoffset), mdY4tw(rawslope4trackletword), mY4tw(rawoffset4trackletword)
{
  // constructor
  mGeo = Geometry::instance();
}

Tracklet::Tracklet(const Tracklet& rhs) : mHCId(rhs.mHCId), mMCM(rhs.mMCM), mROB(rhs.mROB), mQ0(rhs.mQ0), mQ1(rhs.mQ1), mPID(rhs.mPID), mdY(rhs.mdY), mY(rhs.mY), mdY4tw(rhs.mdY4tw), mY4tw(rhs.mY4tw), mNHits(rhs.mNHits), mNHits0(rhs.mNHits0), mNHits1(rhs.mNHits1), mSlope(rhs.mSlope), mOffset(rhs.mOffset), mError(rhs.mError), mNClusters(rhs.mNClusters)
{
  // copy constructor
  mGeo = Geometry::instance();

  mResiduals = rhs.mResiduals;
  mClsCharges = rhs.mClsCharges;
}

unsigned int Tracklet::getTrackletWord() const
{

  int rndAdd = 0;
  int decPlaces = 5; // must be larger than 1 or change the following code
                     // if (decPlaces >  1)
  rndAdd = (1 << (decPlaces - 1)) + 1;
  uint32_t trackletWord;
  unsigned int pid, padrow, slope, offset;

  pid = ((int)mPID) & 0xff;     //8bits
  padrow = ((int)getZ()) & 0xf; //4bits
  slope = ((int)mdY) & 0x7ff;   //7bits
  offset = ((int)mY) & 0x1fff;  //13bits
                                //mMCMT[cpu] = (pid << 24) | (padrow << 20) | (slope << 13) | offset;
  trackletWord = (pid << 24) | (padrow << 20) | (slope << 13) | offset;

  return trackletWord;
}

int Tracklet::getYbin() const
{
  return mY;
}

float Tracklet::getdY() const
{
  return mdY;
}

void Tracklet::setLabel(std::vector<MCCompLabel>& label)
{
  // set the labels
  LOG(info) << "Tracklet set label is disabled labels are stored externally in a MCTruthContainer";
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
