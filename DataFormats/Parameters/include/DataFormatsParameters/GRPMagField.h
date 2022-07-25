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
  o2::units::Current_t getL3Current() const { return mL3Current; }
  o2::units::Current_t getDipoleCurrent() const { return mDipoleCurrent; }
  bool getFieldUniformity() const { return mUniformField; }
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

  ClassDefNV(GRPMagField, 1);
};

} // namespace parameters
} // namespace o2

#endif
