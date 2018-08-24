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
/// \brief Definition of the TOF cluster

#ifndef ALICEO2_TOF_CLUSTER_H
#define ALICEO2_TOF_CLUSTER_H

#include "ReconstructionDataFormats/BaseCluster.h"
#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{
namespace tof
{
/// \class Cluster
/// \brief Cluster class for TOF
///

class Cluster : public o2::BaseCluster<float>
{
 public:
  enum { kMain = 0x3FFFF,
         kUpLeft = 0x40000,     // 2^18, 19th bit
         kUp = 0x80000,         // 2^19, 20th bit
         kUpRight = 0x100000,   // 2^20, 21th bit
         kRight = 0x200000,     // 2^21, 22th bit
         kDownRight = 0x400000, // 2^22, 23th bit
         kDown = 0x800000,      // 2^23, 24th bit
         kDownLeft = 0x1000000, // 2^24, 25th bit
         kLeft = 0x2000000 };   // 2^25, 26th bit

  Cluster() = default;

  Cluster(std::int16_t sensid, float x, float y, float z, float sy2, float sz2, float syz, float timeRaw, float time, float tot, int L0L1latency, int deltaBC);

  ~Cluster() = default;

  float getTimeRaw() const { return mTimeRaw; }             // Cluster ToF getter
  void setTimeRaw(float timeRaw) { mTimeRaw = timeRaw; }    // Cluster ToF setter
  float getTime() const { return mTime; }                   // Cluster ToF getter
  void setTime(float time) { mTime = time; }                // Cluster ToF setter
  float getTot() const { return mTot; }                     // Cluster Charge getter
  void setTot(int tot) { mTot = tot; }                      // Cluster ToT setter
  int getL0L1Latency() const { return mL0L1Latency; };      // L0L1 latency
  void setL0L1Latency(int value) { mL0L1Latency = value; }; // L0-L1 latency
  int getDeltaBC() const { return mDeltaBC; };              // deltaBC
  void setDeltaBC(int value) { mDeltaBC = value; };         // deltaBC
  //float  getZ()   const   {return mZ;}   // Cluster Z - already in the definition of the cluster
  float getR() const { return mR; }     // Cluster Radius
  float getPhi() const { return mPhi; } // Cluster Phi

  int getContributingChannels() const { return mContributingChannels; }
  void setContributingChannels(int contributingChannels) { mContributingChannels = contributingChannels; }
  void addBitInContributingChannels(int mask) { mContributingChannels |= mask; }
  int getNumOfContributingChannels() const; // returns the number of hits associated to the cluster, i.e. the number of hits that built the cluster; it is the equivalente of the old AliESDTOFCluster::GetNTOFhits()
  int getMainContributingChannel() const { return mContributingChannels & kMain; }

  void setMainContributingChannel(int newvalue)
  {
    int mask = kMain;
    mContributingChannels &= ~mask;
    mContributingChannels |= newvalue & mask;
  } // first we do "bitwise-and" with all bits set to 1 but the first 18, which is the negation of "mask"

  // setting the up, down, right, left bits
  void setUpLeftContributingChannel() { mContributingChannels |= kUpLeft; }       // we do "bitwise-or" with the 19th bit (2^18)
  void setUpContributingChannel() { mContributingChannels |= kUp; }               // we do "bitwise-or" with the 20th bit (2^19)
  void setUpRightContributingChannel() { mContributingChannels |= kUpRight; }     // we do "bitwise-or" with the 21th bit (2^20)
  void setRightContributingChannel() { mContributingChannels |= kRight; }         // we do "bitwise-or" with the 22th bit (2^21)
  void setDownRightContributingChannel() { mContributingChannels |= kDownRight; } // we do "bitwise-or" with the 23th bit (2^22)
  void setDownContributingChannel() { mContributingChannels |= kDown; }           // we do "bitwise-or" with the 24th bit (2^23)
  void setDownLeftContributingChannel() { mContributingChannels |= kDownLeft; }   // we do "bitwise-or" with the 25th bit (2^24)
  void setLeftContributingChannel() { mContributingChannels |= kLeft; }           // we do "bitwise-or" with the 26th bit (2^25)

  // resetting the up, down, right, left bits
  void resetUpLeftContributingChannel() { mContributingChannels &= ~kUpLeft; }       // we do "bitwise-and" with the 19th bit only set to zero, which is the negation of "mask"
  void resetUpContributingChannel() { mContributingChannels &= ~kUp; }               // we do "bitwise-and" with the 20th bit only set to zero, which is the negation of "mask"
  void resetUpRightContributingChannel() { mContributingChannels &= ~kUpRight; }     // we do "bitwise-and" with the 21th bit only set to zero, which is the negation of "mask"
  void resetRightContributingChannel() { mContributingChannels &= ~kRight; }         // we do "bitwise-and" with the 22th bit only set to zero, which is the negation of "mask"
  void resetDownRightContributingChannel() { mContributingChannels &= ~kDownRight; } // we do "bitwise-and" with the 23th bit only set to zero, which is the negation of "mask"
  void resetDownContributingChannel() { mContributingChannels &= ~kDown; }           // we do "bitwise-and" with the 24th bit only set to zero, which is the negation of "mask"
  void resetDownLeftContributingChannel() { mContributingChannels &= ~kDownLeft; }   // we do "bitwise-and" with the 25th bit only set to zero, which is the negation of "mask"
  void resetLeftContributingChannel() { mContributingChannels &= ~kLeft; }           // we do "bitwise-and" with the 26th bit only set to zero, which is the negation of "mask"

  // getters of the up, down, right, left bits
  bool isUpLeftContributing() const { return mContributingChannels & kUpLeft; }
  bool isUpContributing() const { return mContributingChannels & kUp; }
  bool isUpRightContributing() const { return mContributingChannels & kUpRight; }
  bool isRightContributing() const { return mContributingChannels & kRight; }
  bool isDownRightContributing() const { return mContributingChannels & kDownRight; }
  bool isDownContributing() const { return mContributingChannels & kDown; }
  bool isDownLeftContributing() const { return mContributingChannels & kDownLeft; }
  bool isLeftContributing() const { return mContributingChannels & kLeft; }

 private:
  friend class boost::serialization::access;

  float mTimeRaw;   // raw TOF time // CZ: in AliRoot it is a double
  float mTime;      // calibrated TOF time // CZ: in AliRoot it is a double
  float mTot;       // Time-Over-threshold // CZ: in AliRoot it is a double
  int mL0L1Latency; // L0L1 latency // CZ: is it different per cluster? Checking one ESD file, it seems that it is always the same (see: /alice/data/2017/LHC17n/000280235/pass1/17000280235019.100/AliESDs.root)
  int mDeltaBC;     // DeltaBC --> can it be a char or short? // CZ: is it different per cluster? Checking one ESD file, it seems that it can vary (see: /alice/data/2017/LHC17n/000280235/pass1/17000280235019.100/AliESDs.root)
  //float  mZ;           //! z-coordinate // CZ: to be verified if it is the same in the BaseCluster class
  float mR;                  //! radius
  float mPhi;                //! phi coordinate
  int mContributingChannels; // index of the channels that contributed to the cluster; to be read like this:
                             // channel & 0x3FFFF -> first 18 bits to store the main channel
                             // channel & bit19 (0x40000) -> alsoUPLEFT
                             // channel & bit20 (0x80000) -> alsoUP
                             // channel & bit21 (0x100000)-> alsoUPRIGHT
                             // channel & bit22 (0x200000)-> alsoRIGHT
                             // channel & bit23 (0x400000)-> alsoDOWNRIGHT
                             // channel & bit24 (0x800000)-> alsoDOWN
                             // channel & bit25 (0x1000000)-> alsoDOWNLEFT
                             // channel & bit26 (0x2000000)-> alsoLEFT

  ClassDefNV(Cluster, 1);
};

std::ostream& operator<<(std::ostream& os, const Cluster& c);
} // namespace TOF
} // namespace o2
#endif
