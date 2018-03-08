// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  MpArea(float x1 = 0., float y1 = 0., float x2 = 0., float y2 = 0.);
  // virtual ~MpArea() = default;

  float getCenterX() const;
  float getCenterY() const;
  float getHalfSizeX() const;
  float getHalfSizeY() const;
  bool isValid() const;

  /// Set x min
  void setXmin(float val) { mPositions[0] = val; }
  /// Set x max
  void setXmax(float val) { mPositions[2] = val; }
  /// Set y min
  void setYmin(float val) { mPositions[1] = val; }
  /// Set y max
  void setYmax(float val) { mPositions[3] = val; }

  /// Get x min
  float getXmin() const { return mPositions[0]; }
  /// Get x max
  float getXmax() const { return mPositions[2]; }
  /// Get y min
  float getYmin() const { return mPositions[1]; }
  /// Get y max
  float getYmax() const { return mPositions[3]; }

 private:
  std::array<float, 4> mPositions;
};
} // namespace mid
} // namespace o2
#endif /* O2_MID_MPAREA_H */
