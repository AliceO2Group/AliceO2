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

/// \file GRPMagField.h
/// \brief Header of the General Run Parameters object for B field values
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_DATA_GRPMAGFIELDOBJECT_H_
#define ALICEO2_DATA_GRPMAGFIELDOBJECT_H_

#include <Rtypes.h>
#include "CommonTypes/Units.h"

namespace o2
{
namespace parameters
{
/*
 * Collects parameters describing the run that are related to the B field only.
*/

class GRPMagField
{
 public:
  GRPMagField() = default;
  ~GRPMagField() = default;

  /// getters/setters for magnets currents
  o2::units::Current_t getL3Current() const
  {
    static float v = checkL3Override();
    return v == NOOVERRIDEVAL ? mL3Current : v;
  }

  o2::units::Current_t getDipoleCurrent() const
  {
    static float v = checkDipoleOverride();
    return v == NOOVERRIDEVAL ? mDipoleCurrent : v;
  }

  bool getFieldUniformity() const { return mUniformField; }
  int8_t getNominalL3Field();
  void setL3Current(o2::units::Current_t v) { mL3Current = v; }
  void setDipoleCurrent(o2::units::Current_t v) { mDipoleCurrent = v; }
  void setFieldUniformity(bool v) { mUniformField = v; }

  /// print itself
  void print() const;

  static GRPMagField* loadFrom(const std::string& grpMagFieldFileName = "");

 private:
  o2::units::Current_t mL3Current = 0.f;     ///< signed current in L3
  o2::units::Current_t mDipoleCurrent = 0.f; ///< signed current in Dipole
  bool mUniformField = false;                ///< uniformity of magnetic field
  int8_t mNominalL3Field = 0;                //!< Nominal L3 field deduced from mL3Current
  bool mNominalL3FieldValid = false;         //!< Has the field been computed (for caching)

  static constexpr float NOOVERRIDEVAL = 1e99;
  static float checkL3Override();
  static float checkDipoleOverride();

  ClassDefNV(GRPMagField, 2);
};

inline int8_t GRPMagField::getNominalL3Field()
{
  // compute nominal L3 field in kG

  if (mNominalL3FieldValid == false) {
    mNominalL3Field = std::lround(5.f * getL3Current() / 30000.f);
    mNominalL3FieldValid = true;
  }
  return mNominalL3Field;
}

} // namespace parameters
} // namespace o2

#endif
