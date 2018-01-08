// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_HIT_H
#define ALICEO2_PHOS_HIT_H

#include "SimulationDataFormat/BaseHits.h"

namespace o2
{
namespace PHOS
{
/// \class Hit
/// \brief PHOS simulation hit information
class Hit : public o2::BasicXYZEHit<float>
{
 public:
  /// \brief Default constructor
  Hit() = default;

  /// \brief Hit constructor
  ///
  /// Fully defining information of the PHOS point (position,
  /// momentum, energy, track, ...)
  ///
  /// \param trackID Index of the track entered PHOS
  /// \param detID ID of the detector segment
  /// \param pos Position vector of the point
  /// \param mom Momentum vector for the particle at the point
  /// \param initialEnergy Energy of the primary particle enering the EMCAL
  /// \param tof Time of the hit
  /// \param length Length of the segment
  Hit(Int_t trackID, Int_t detID, const Point3D<float>& pos,
      const Vector3D<float>& mom, Double_t totE, Double_t tof, Double_t eLoss)
    : o2::BasicXYZEHit<float>(pos.X(), pos.Y(), pos.Z(), tof, eLoss, trackID, detID),
      mPvector(mom),
      mInitialEnergy(totE)
  {
  }

  /// \brief Check whether the points are from the same parent and in the same detector volume
  /// \return True if points are the same (origin and detector), false otherwise
  Bool_t operator==(const Hit& rhs) const;

  /// \brief Sorting points according to parent particle and detector volume
  /// \return True if this point is smaller, false otherwise
  Bool_t operator<(const Hit& rhs) const;

  /// \brief Adds energy loss from the other point to this point
  /// \param rhs EMCAL point to add to this point
  /// \return This point with the summed energy loss
  Hit& operator+=(const Hit& rhs);

  /// \brief Creates a new point base on this point but adding the energy loss of the right hand side
  /// \param
  /// \return New EMAL point base on this point
  Hit operator+(const Hit& rhs) const;

  static bool CompareAndAdd(Hit &a, const Hit &b){ if(a==b){a+=b; return true ;} else return false ; }

  /// \brief Destructor
  ~Hit() = default;

  /// \brief Get the initial energy of the primary particle entering EMCAL
  /// \return Energy of the primary particle entering EMCAL
  Double_t GetInitialEnergy() const { return mInitialEnergy; }


  void AddEnergyLoss(Double_t eloss){SetEnergyLoss(GetEnergyLoss()+eloss) ; }
 
  /// \brief Writing point information to an output stream;
  /// \param stream target output stream
  void PrintStream(std::ostream& stream) const;

 private:
  Vector3D<float> mPvector;  ///< Momentum Vector
  Double32_t mInitialEnergy; ///< Energy of the parent particle that entered the PHOS front surface

  ClassDefNV(Hit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Hit& point);
}
}

#endif /* Point_h */
