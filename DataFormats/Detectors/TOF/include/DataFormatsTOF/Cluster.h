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

#include "GPUCommonRtypes.h"
#include "GPUCommonMath.h"
#include "ReconstructionDataFormats/BaseCluster.h"
#include "CommonConstants/LHCConstants.h"
#ifndef GPUCA_GPUCODE
#include <boost/serialization/base_object.hpp> // for base_object
#include <cstdlib>
#include <vector>
#endif

namespace o2
{
namespace tof
{
/// \class Cluster
/// \brief Cluster class for TOF
///

class Cluster : public o2::BaseCluster<float>
{
  static constexpr float RadiusOutOfRange = 9999; // used to check if the radius was already calculated or not
  static constexpr float PhiOutOfRange = 9999;    // used to check if phi was already calculated or not

  static constexpr int NPADSXSECTOR = 8736;
  static constexpr double BC_TIME_INPS_INV = 1.E-3 / o2::constants::lhc::LHCBunchSpacingNS;

 public:
  enum { kUpLeft = 0,    // 2^0, 1st bit
         kUp = 1,        // 2^1, 2nd bit
         kUpRight = 2,   // 2^2, 3rd bit
         kRight = 3,     // 2^3, 4th bit
         kDownRight = 4, // 2^4, 5th bit
         kDown = 5,      // 2^5, 6th bit
         kDownLeft = 6,  // 2^6, 7th bit
         kLeft = 7 };    // 2^7, 8th bit

  Cluster() = default;

  Cluster(std::int16_t sensid, float x, float y, float z, float sy2, float sz2, float syz, double timeRaw, double time, float tot, int L0L1latency, int deltaBC);

  ~Cluster() = default;

  std::int8_t getSector() const { return getCount(); }
  void setSector(std::int8_t value) { setCount(value); }

  std::int16_t getPadInSector() const { return getSensorID(); }
  void setPadInSector(std::int16_t value) { setSensorID(value); }

  double getTimeRaw() const { return mTimeRaw; }            // Cluster ToF getter
  void setTimeRaw(double timeRaw) { mTimeRaw = timeRaw; }   // Cluster ToF setter
  double getTime() const { return mTime; }                  // Cluster ToF getter
  void setTime(double time) { mTime = time; }               // Cluster ToF setter
  float getTot() const { return mTot; }                     // Cluster Charge getter
  void setTot(float tot) { mTot = tot; }                    // Cluster ToT setter
  int getL0L1Latency() const { return mL0L1Latency; };      // L0L1 latency
  void setL0L1Latency(int value) { mL0L1Latency = value; }; // L0-L1 latency
  int getDeltaBC() const { return mDeltaBC; };              // deltaBC
  void setDeltaBC(int value) { mDeltaBC = value; };         // deltaBC
  //float  getZ()   const   {return mZ;}   // Cluster Z - already in the definition of the cluster

  float getR() // Cluster Radius (it is the same in sector and global frame)
  {
    if (mR == RadiusOutOfRange) {
      mR = o2::gpu::CAMath::Sqrt(getX() * getX() + getY() * getY() + getZ() * getZ());
    }
    return mR;
  }

  float getPhi() // Cluster Phi in sector frame
  {
    if (mPhi == PhiOutOfRange) {
      mPhi = o2::gpu::CAMath::ATan2(getY(), getX());
    }
    return mPhi;
  }
  float getR() const // Cluster Radius (it is the same in sector and global frame)
  {
    if (mR == RadiusOutOfRange) {
      return o2::gpu::CAMath::Sqrt(getX() * getX() + getY() * getY() + getZ() * getZ());
    }
    return mR;
  }

  float getPhi() const // Cluster Phi in sector frame
  {
    if (mPhi == PhiOutOfRange) {
      return o2::gpu::CAMath::ATan2(getY(), getX());
    }
    return mPhi;
  }

  void setR(float value) { mR = value; }
  void setPhi(float value) { mPhi = value; }

  int getNumOfContributingChannels() const; // returns the number of hits associated to the cluster, i.e. the number of hits that built the cluster; it is the equivalente of the old AliESDTOFCluster::GetNTOFhits()
  int getMainContributingChannel() const { return getSector() * NPADSXSECTOR + getPadInSector(); }

  void addBitInContributingChannels(int bit) { setBit(bit); }
  void resetBitInContributingChannels(int bit) { resetBit(bit); }
  std::uint8_t getAdditionalContributingChannels() const { return getBits(); }
  void setAdditionalContributingChannels(std::uint8_t mask) { setBits(mask); }

  bool isAdditionalChannelSet(int bit /* e.g. o2::tof::Cluster::kUpLeft */) const { return isBitSet(bit); }

  void setMainContributingChannel(int newvalue)
  {
    setSector(newvalue / NPADSXSECTOR);
    setPadInSector(newvalue % NPADSXSECTOR);
  }

  void setEntryInTree(int value) { mEntryInTree = value; }
  int getEntryInTree() const { return mEntryInTree; }

  int getBC() const { return int(mTimeRaw * BC_TIME_INPS_INV); }

  void setDigitInfo(int idig, int ch, double t, float tot);
  int getDigitInfoCH(int idig) const { return mDigitInfoCh[idig]; }
  double getDigitInfoT(int idig) const { return mDigitInfoT[idig]; }
  float getDigitInfoTOT(int idig) const { return mDigitInfoTOT[idig]; }

 private:
  friend class boost::serialization::access;

  double mTimeRaw;  // raw TOF time // CZ: in AliRoot it is a double
  double mTime;     // calibrated TOF time // CZ: in AliRoot it is a double
  float mTot;       // Time-Over-threshold // CZ: in AliRoot it is a double
  int mL0L1Latency; // L0L1 latency // CZ: is it different per cluster? Checking one ESD file, it seems that it is always the same (see: /alice/data/2017/LHC17n/000280235/pass1/17000280235019.100/AliESDs.root)
  int mDeltaBC;     // DeltaBC --> can it be a char or short? // CZ: is it different per cluster? Checking one ESD file, it seems that it can vary (see: /alice/data/2017/LHC17n/000280235/pass1/17000280235019.100/AliESDs.root)
  //float  mZ;           //! z-coordinate // CZ: to be verified if it is the same in the BaseCluster class
  float mR = RadiusOutOfRange; //! radius
  float mPhi = PhiOutOfRange;  //! phi coordinate
  int mEntryInTree;            //! index of the entry in the tree from which we read the cluster

  // add extra info to trace all digit infos (for commissioning phase)
  int mDigitInfoCh[6] = {0, 0, 0, 0, 0, 0};
  double mDigitInfoT[6] = {0., 0., 0., 0., 0., 0.};
  float mDigitInfoTOT[6] = {0., 0., 0., 0., 0., 0.};

  ClassDefNV(Cluster, 4);
};

#ifndef GPUCA_GPUCODE
std::ostream& operator<<(std::ostream& os, Cluster& c);
#endif
} // namespace tof

/// Defining o2::tof::Cluster explicitly as messageable
///
/// o2::tof::Cluster does not fulfill is_messageable because the underlying ROOT
/// classes of Point3D are note trivially copyable.
/// std::is_trivially_copyable<ROOT::Math::Cartesian3D<float>> fails because the class
/// implements a copy constructor, although it does not much more than the default copy
/// constructor. Have been trying to specialize std::is_trivially_copyable for Point3D
/// alias in MathUtils/Cartesian.h, but structures with a member of Point3D are
/// still not fulfilling the condition. Need to understand how the type trait checks
/// the condition for members.
/// We believe that o2::tof::Cluster is messageable and explicitly specialize the
/// type trait, adding a corresponding unit test to go beyond make-believe
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::tof::Cluster> : std::true_type {
};
} // namespace framework

} // namespace o2
#endif
