// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ITS3_ALIGNMENT_LABEL_H
#define O2_ITS3_ALIGNMENT_LABEL_H

#include <cstdint>
#include <string>
#include <format>

class GlobalLabel
{
  // Millepede label is any positive integer [1....)
  // Layout: DOF(5) | CALIB(1) | ID(22) | SENS(1) | DET(2) = 31 usable bits (MSB reserved, GBL uses signed int)
 public:
  using T = uint32_t;
  static constexpr int DOF_BITS = 5;   // bits 0-4
  static constexpr int CALIB_BITS = 1; // bit 5: 0 = rigid body, 1 = calibration (only allow for one calibration, could be extended if needed)
  static constexpr int ID_BITS = 22;   // bits 6-27
  static constexpr int SENS_BITS = 1;  // bit 28
  static constexpr int TOTAL_BITS = sizeof(T) * 8;
  static constexpr int DET_BITS = TOTAL_BITS - (DOF_BITS + CALIB_BITS + ID_BITS + SENS_BITS) - 1; // one less bit since GBL uses int!
  static constexpr T bitMask(int b) noexcept
  {
    return (T(1) << b) - T(1);
  }
  static constexpr int DOF_SHIFT = 0;
  static constexpr T DOF_MAX = (T(1) << DOF_BITS) - T(1);
  static constexpr T DOF_MASK = DOF_MAX << DOF_SHIFT;
  static constexpr int CALIB_SHIFT = DOF_BITS;
  static constexpr T CALIB_MAX = (T(1) << CALIB_BITS) - T(1);
  static constexpr T CALIB_MASK = CALIB_MAX << CALIB_SHIFT;
  static constexpr int ID_SHIFT = DOF_BITS + CALIB_BITS;
  static constexpr T ID_MAX = (T(1) << ID_BITS) - T(1);
  static constexpr T ID_MASK = ID_MAX << ID_SHIFT;
  static constexpr int SENS_SHIFT = DOF_BITS + CALIB_BITS + ID_BITS;
  static constexpr T SENS_MAX = (T(1) << SENS_BITS) - T(1);
  static constexpr T SENS_MASK = SENS_MAX << SENS_SHIFT;
  static constexpr int DET_SHIFT = DOF_BITS + CALIB_BITS + ID_BITS + SENS_BITS;
  static constexpr T DET_MAX = (T(1) << DET_BITS) - T(1);
  static constexpr T DET_MASK = DET_MAX << DET_SHIFT;

  GlobalLabel(T det, T id, bool sens, bool calib = false)
    : mID((((id + 1) & ID_MAX) << ID_SHIFT) |
          ((det & DET_MAX) << DET_SHIFT) |
          ((T(sens) & SENS_MAX) << SENS_SHIFT) |
          ((T(calib) & CALIB_MAX) << CALIB_SHIFT))
  {
  }

  /// produce the raw Millepede label for a given DOF index (rigid body: calib=0 in label)
  constexpr T raw(T dof) const noexcept { return (mID & ~DOF_MASK) | ((dof & DOF_MAX) << DOF_SHIFT); }
  constexpr int rawGBL(T dof) const noexcept { return static_cast<int>(raw(dof)); }

  /// return a copy of this label with the CALIB bit set (for calibration DOFs on same volume)
  GlobalLabel asCalib() const noexcept
  {
    GlobalLabel c{*this};
    c.mID |= (T(1) << CALIB_SHIFT);
    return c;
  }

  constexpr T id() const noexcept { return ((mID >> ID_SHIFT) & ID_MAX) - 1; }
  constexpr T det() const noexcept { return (mID & DET_MASK) >> DET_SHIFT; }
  constexpr bool sens() const noexcept { return (mID & SENS_MASK) >> SENS_SHIFT; }
  constexpr bool calib() const noexcept { return (mID & CALIB_MASK) >> CALIB_SHIFT; }

  std::string asString() const
  {
    return std::format("Det:{} Id:{} Sens:{} Calib:{}", det(), id(), sens(), calib());
  }

  constexpr auto operator<=>(const GlobalLabel&) const noexcept = default;

 private:
  T mID{0};
};

#endif
