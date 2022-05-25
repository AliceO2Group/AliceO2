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

#ifndef ALICEO2_CPV_HIT_H
#define ALICEO2_CPV_HIT_H

#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace cpv
{
/// \class Hit
/// \brief CPV simulation hit information
class Hit : public o2::BasicXYZEHit<float>
{
 public:
  /// \brief Default constructor
  Hit() = default;

  /// \brief Hit constructor
  ///
  /// Fully defining information of the CPV Hit (position,
  /// momentum, energy, track, ...)
  ///
  /// \param trackID Index of the track entered CPV
  /// \param detID ID of the detector segment
  /// \param pos Position vector of the Hit
  /// \param mom Momentum vector for the particle at the Hit
  /// \param initialEnergy Energy of the primary particle enering the EMCAL
  /// \param tof Time of the hit
  /// \param length Length of the segment
  Hit(int trackID, int detID, const math_utils::Point3D<float>& pos, double tof, double qLoss)
    : o2::BasicXYZEHit<float>(pos.X(), pos.Y(), pos.Z(), tof, qLoss, trackID, detID)
  {
  }

  Hit& operator=(const Hit& hit) = default;

  /// \brief Check whether the points are from the same SuperParent and in the same detector volume
  /// \return True if points are the same (origin and detector), false otherwise
  Bool_t operator==(const Hit& rhs) const;

  /// \brief Sorting points according to parent particle and detector volume
  /// \return True if this Hit is smaller, false otherwise
  Bool_t operator<(const Hit& rhs) const;

  /// \brief Adds energy loss from the other Hit to this Hit
  /// \param rhs cpv::Hit to add to this Hit
  /// \return This Hit with the summed energy loss
  Hit& operator+=(const Hit& rhs);

  /// \brief Creates a new Hit based on this Hit but adding the energy loss of the right hand side
  /// \param
  /// \return New Hit based on this Hit
  // Hit operator+(const Hit& rhs) const;
  friend Hit operator+(const Hit& lhs, const Hit& rhs);

  /// \brief Destructor
  ~Hit() = default;

  void AddEnergyLoss(Double_t eloss) { SetEnergyLoss(GetEnergyLoss() + eloss); }

  /// \brief Writing Hit information to an output stream;
  /// \param stream target output stream
  void PrintStream(std::ostream& stream) const;

 private:
  ClassDefNV(Hit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Hit& point);
} // namespace cpv
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::cpv::Hit> : public o2::utils::ShmAllocator<o2::cpv::Hit>
{
};
} // namespace std
#endif

#endif /* Hit_h */
