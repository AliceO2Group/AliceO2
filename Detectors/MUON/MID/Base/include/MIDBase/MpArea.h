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

/// \file   MIDBase/MpArea.h
/// \brief  Mapping area for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 December 2017
#ifndef O2_MID_MPAREA_H
#define O2_MID_MPAREA_H

#include <array>

namespace o2
{
namespace mid
{
class MpArea
{
 public:
  MpArea(double x1 = 0., double y1 = 0., double x2 = 0., double y2 = 0.);
  // virtual ~MpArea() = default;

  double getCenterX() const;
  double getCenterY() const;
  double getHalfSizeX() const;
  double getHalfSizeY() const;
  bool isValid() const;

  /// Set x min
  void setXmin(double val) { mPositions[0] = val; }
  /// Set x max
  void setXmax(double val) { mPositions[2] = val; }
  /// Set y min
  void setYmin(double val) { mPositions[1] = val; }
  /// Set y max
  void setYmax(double val) { mPositions[3] = val; }

  /// Get x min
  double getXmin() const { return mPositions[0]; }
  /// Get x max
  double getXmax() const { return mPositions[2]; }
  /// Get y min
  double getYmin() const { return mPositions[1]; }
  /// Get y max
  double getYmax() const { return mPositions[3]; }

 private:
  std::array<double, 4> mPositions;
};
} // namespace mid
} // namespace o2
#endif /* O2_MID_MPAREA_H */
