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

/// \file Hit.h
/// \brief MC hit class to store energy loss per cell and per superparent
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_HIT_H
#define ALICEO2_ECAL_HIT_H

#include <SimulationDataFormat/BaseHits.h>
#include <CommonUtils/ShmAllocator.h>

namespace o2
{
namespace ecal
{
class Hit : public o2::BasicXYZEHit<float>
{
 public:
  /// \brief Default constructor
  Hit() = default;

  /// \brief Hit constructor
  ///
  /// Fully defining information of the ECAL point (position, momentum, energy, track, ...)
  ///
  /// \param trackID Index of the track, defined as parent track entering the ECAL
  /// \param cellID ID of the detector cell
  /// \param pos Position vector of the point
  /// \param mom Momentum vector for the particle at the point
  /// \param tof Time of the hit
  /// \param eLoss Energy loss
  Hit(int trackID, int cellID, const math_utils::Point3D<float>& pos,
      const math_utils::Vector3D<float>& mom, float tof, float eLoss)
    : o2::BasicXYZEHit<float>(pos.X(), pos.Y(), pos.Z(), tof, eLoss, trackID, 0),
      mPvector(mom),
      mCellID(cellID)
  {
  }

  /// \brief Destructor
  ~Hit() = default;

  /// \brief Check whether the points are from the same parent and in the same detector volume
  /// \return True if points are the same (origin and detector), false otherwise
  bool operator==(const Hit& rhs) const;

  /// \brief Sorting points according to parent particle and detector volume
  /// \return True if this point is smaller, false otherwise
  bool operator<(const Hit& rhs) const;

  /// \brief Get cell ID
  /// \return cell ID
  int GetCellID() const { return mCellID; }

 private:
  math_utils::Vector3D<float> mPvector; ///< Momentum vector
  int32_t mCellID;                      ///< Cell ID (used instead of short detID)
  ClassDefNV(Hit, 1);
};

} // namespace ecal
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::ecal::Hit> : public o2::utils::ShmAllocator<o2::ecal::Hit>
{
};
} // namespace std
#endif

#endif // ALICEO2_ECAL_HIT_H
