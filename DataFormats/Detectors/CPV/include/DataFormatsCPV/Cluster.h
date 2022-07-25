// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_CLUSTER_H_
#define ALICEO2_CPV_CLUSTER_H_

#include "DataFormatsCPV/Digit.h"
#include <cmath>

namespace o2
{
namespace cpv
{
class Geometry;
/// \class Cluster
/// \brief Contains CPV cluster parameters

constexpr float kMinX = -72.32;  // Minimal coordinate in X direction
constexpr float kStepX = 0.0025; // digitization step in X direction
constexpr float kMinZ = -63.3;   // Minimal coordinate in Z direction
constexpr float kStepZ = 0.002;  // digitization step in Z direction
constexpr float kStepE = 0.036;  // Amplitude digitization parameter

class Cluster
{

  union CluStatus {
    uint8_t mBits;
    struct {
      uint8_t multiplicity : 5; // Pad multiplicty, bits 0-4
      uint8_t module : 2;       // module number, bits 5-6
      uint8_t unfolded : 1;     // unfolded bit, bit 7
    };
  };

 public:
  Cluster() = default;
  Cluster(unsigned char mult, char mod, char exMax, float x, float z, float e) : mMulDigit(mult), mModule(mod), mNExMax(exMax), mLocalPosX(x), mLocalPosZ(z), mEnergy(e) {}
  Cluster(const Cluster& clu) = default;

  ~Cluster() = default;

  /// \brief Comparison oparator, based on time and coordinates
  /// \param another CPV Cluster
  /// \return result of comparison: x and z coordinates
  bool operator<(const Cluster& other) const;
  /// \brief Comparison oparator, based on time and coordinates
  /// \param another CPV Cluster
  /// \return result of comparison: x and z coordinates
  bool operator>(const Cluster& other) const;

  void setEnergy(float e) { mEnergy = e; }
  float getEnergy() const { return mEnergy; }

  void getLocalPosition(float& posX, float& posZ) const
  {
    posX = mLocalPosX;
    posZ = mLocalPosZ;
  }
  unsigned char getMultiplicity() const { return mMulDigit; } // gets the number of digits making this recpoint
                                                              // 0: was no unfolging, -1: unfolding failed
  char getModule() const { return mModule; }                  // CPV module of a current cluster

  // 0: was no unfolging, -1: unfolding failed
  void setNExMax(char nmax = 1) { mNExMax = nmax; }
  char getNExMax() const { return mNExMax; } // Number of maxima found in cluster in unfolding:
                                             // 0: was no unfolging, -1: unfolding failed

  // raw access for CTF encoding
  uint16_t getPackedPosX() const { return uint16_t((mLocalPosX - kMinX) / kStepX); }
  void setPackedPosX(uint16_t v) { mLocalPosX = kMinX + kStepX * v; }

  uint16_t getPackedPosZ() const { return uint16_t((mLocalPosZ - kMinZ) / kStepZ); }
  void setPackedPosZ(uint16_t v) { mLocalPosZ = kMinZ + kStepZ * v; }

  uint8_t getPackedEnergy() const { return uint8_t(std::min(255, int((mEnergy > 100.) ? (log(mEnergy - 63.) / kStepE) : mEnergy))); }
  void setPackedEnergy(uint8_t v) { mEnergy = ((v > 100) ? (exp(kStepE * v) + 63.) : (v * 1.)); }

  uint8_t getPackedClusterStatus() const
  {
    CluStatus s = {0};
    s.multiplicity = std::min(mMulDigit, static_cast<unsigned char>(31)); // 5 bits available
    s.module = mModule - 2;
    s.unfolded = mNExMax > 1;
    return s.mBits;
  }
  void setPackedClusterStatus(uint8_t v)
  {
    CluStatus s = {v};
    mMulDigit = s.multiplicity;
    mModule = s.module + 2;
    mNExMax = s.unfolded ? 1 : 2;
  }

  void setPacked(uint16_t posX, uint16_t posZ, uint8_t en, uint8_t status)
  {
    setPackedPosX(posX);
    setPackedPosZ(posZ);
    setPackedEnergy(en);
    setPackedClusterStatus(status);
  }

 protected:
  unsigned char mMulDigit = 0; ///< Digit nultiplicity
  char mModule = 0;            ///< Module number
  char mNExMax = -1;           ///< number of (Ex-)maxima before unfolding
  float mLocalPosX = 0.;       ///< Center of gravity position in local module coordunates (phi direction)
  float mLocalPosZ = 0.;       ///< Center of gravity position in local module coordunates (z direction)
  float mEnergy = 0.;          ///< full energy of a cluster

  ClassDefNV(Cluster, 1);
};

} // namespace cpv

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::cpv::Cluster> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif
