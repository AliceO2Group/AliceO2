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

/// \file SegmentationSuperAlpide.h
/// \brief Definition of the SegmentationSuperAlpide class
/// \author Fabrizio Grosa <fgrosa@cern.ch>
/// \author felix.schlepper@cern.ch

#ifndef ALICEO2_ITS3_SEGMENTATIONSUPERALPIDE_H_
#define ALICEO2_ITS3_SEGMENTATIONSUPERALPIDE_H_

#include "MathUtils/Cartesian.h"
#include "CommonConstants/MathConstants.h"
#include "ITSBase/Specs.h"

namespace o2
{
namespace its3
{

/// Segmentation and response for pixels in ITS3 upgrade
template <typename value_t = double>
class SegmentationSuperAlpide
{
  // This class defines the segmenation of the pixelArray in the tile. We define
  // two coordinate systems, one width x,z detector local coordianates (cm) and
  // the more natural row,col layout: Also all the transformation between these
  // two. The class provides the transformation from the tile to TGeo
  // coordinates.

  // row,col=0
  // |
  // v
  // x----------------------x
  // |           |          |
  // |           |          |
  // |           |          |
  // |           |          |                        ^ x
  // |           |          |                        |
  // |           |          |                        |
  // |           |          |                        |
  // |-----------X----------|  X marks (x,z)=(0,0)   X----> z
  // |           |          |
  // |           |          |
  // |           |          |
  // |           |          |
  // |           |          |
  // |           |          |
  // x----------------------x

 public:
  SegmentationSuperAlpide(int layer = 0) : mLayer{layer} {}

  /// Transformation from the curved surface to a flat surface
  /// It works only if the detector is not rototraslated
  /// \param xCurved Detector local curved coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yCurved Detector local curved coordinate y in cm with respect to
  /// the center of the sensitive volume.
  /// \param xFlat Detector local flat coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yFlat Detector local flat coordinate y in cm with respect to
  /// the center of the sensitive volume.
  void curvedToFlat(value_t xCurved, value_t yCurved, value_t& xFlat, value_t& yFlat)
  {
    value_t dist = std::hypot(xCurved, yCurved);
    yFlat = dist - mEffRadius;
    value_t phi = (vaule_t)constants::math::PI / 2 - std::atan2((double)yCurved, (double)xCurved);
    xFlat = mEffRadius * phi;
  }

  /// Transformation from the flat surface to a curved surface
  /// It works only if the detector is not rototraslated
  /// \param xFlat Detector local flat coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yFlat Detector local flat coordinate y in cm with respect to
  /// the center of the sensitive volume.
  /// \param xCurved Detector local curved coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yCurved Detector local curved coordinate y in cm with respect to
  /// the center of the sensitive volume.
  void flatToCurved(value_t xFlat, value_t yFlat, value_t& xCurved, value_t& yCurved)
  {
    value_t dist = yFlat + mEffRadius;
    value_t phi = xFlat / dist;
    value_t tang = std::tan((value_t)constants::math::PI / 2 - (value_t)phi);
    xCurved = (xFlat > 0 ? 1.f : -1.f) * dist / std::sqrt(1 + tang * tang);
    yCurved = xCurved * tang;
  }

  /// Transformation from Geant detector centered local coordinates (cm) to
  /// Pixel cell numbers iRow and iCol.
  /// Returns true if point x,z is inside sensitive volume, false otherwise.
  /// A value of -1 for iRow or iCol indicates that this point is outside of the
  /// detector segmentation as defined.
  /// \param float x Detector local coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param float z Detector local coordinate z in cm with respect to
  /// the center of the sensitive volume.
  /// \param int iRow Detector x cell coordinate.
  /// \param int iCol Detector z cell coordinate.
  bool localToDetector(value_t const xRow, value_t const zCol, int& iRow, int& iCol) const noexcept
  {
    localToDetectorUnchecked(xRow, zCol, iRow, iCol);
    if (!isValid(iRow, iCol)) {
      iRow = iCol = -1;
      return false;
    }
    return true;
  }

  // Same as localToDetector w.o. checks.
  void localToDetectorUnchecked(value_t const xRow, value_t const zCol, int& iRow, int& iCol) const noexcept
  {
    namespace cp = constants::pixelarray;
    value_t x = cp::length / 2. - xRow; // transformation to upper edge of pixelarray
    value_t z = zCol + cp::width / 2.;  // transformation to left edge of pixelarray
    iRow = std::floor(x / cp::pixel::pitchRow);
    iCol = std::floor(z / cp::pixel::pitchCol);
  }

  /// Transformation from Detector cell coordinates to Geant detector centered
  /// local coordinates (cm)
  /// \param int iRow Detector x cell coordinate.
  /// \param int iCol Detector z cell coordinate.
  /// \param float x Detector local coordinate x in cm with respect to the
  /// center of the sensitive volume.
  /// \param float z Detector local coordinate z in cm with respect to the
  /// center of the sensitive volume.
  /// If iRow and or iCol is outside of the segmentation range a value of -0.5*Dx()
  /// or -0.5*Dz() is returned.
  bool detectorToLocal(int const iRow, int const iCol, value_t& xRow, value_t& zCol) const noexcept
  {
    detectorToLocalUnchecked(iRow, iCol, xRow, zCol);
    if (!isValid(xRow, zCol)) {
      return false;
    }
    return true;
  }

  // Same as detectorToLocal w.o. checks.
  // We position ourself in the middle of the pixel.
  void detectorToLocalUnchecked(int iRow, int iCol, value_t& xRow, value_t& zCol) const noexcept
  {
    namespace cp = constants::pixelarray;
    xRow = -(iRow - 0.5) * cp::pixel::pitchRow + cp::length / 2.;
    zCol = -(iCol + 0.5) * cp::pixel::pitchCol - cp::width / 2.;
  }

  bool detectorToLocal(float row, float col, float& xRow, float& zCol)
  {
    return detectorToLocal(static_cast<int>(row), static_cast<int>(col), xRow, zCol);
  }

  void detectorToLocalUnchecked(float row, float col, float& xRow, float& zCol)
  {
    detectorToLocalUnchecked(static_cast<int>(row), static_cast<int>(col), xRow, zCol);
  }

  bool detectorToLocal(float row, float col, math_utils::Point3D<float>& loc)
  {
    float xRow, zCol;
    if (detectorToLocal(row, col, xRow, zCol)) {
      return false;
    }
    loc.SetCoordinates(xRow, 0., zCol);
    return true;
  }
  void detectorToLocalUnchecked(float row, float col, math_utils::Point3D<float>& loc)
  {
    float xRow, zCol;
    detectorToLocalUnchecked(row, col, xRow, zCol);
    loc.SetCoordinates(xRow, 0., zCol);
  }

 private:
  bool isValid(value_t const xRow, value_t const zCol) const noexcept
  {
    namespace cp = constants::pixelarray;
    if (xRow < 0. || xRow >= cp::length || zCol < 0. || zCol >= cp::width) {
      return false;
    }
    return true;
  }

  bool isValid(int const iRow, int const iCol) const noexcept
  {
    namespace cp = constants::pixelarray;
    if (iRow < 0 || iRow >= cp::nRows || iCol < 0 || iCol >= cp::nCols) {
      return false;
    }
    return true;
  }

  const int mLayer; ///< chip layer
  const value_t mEffRadius{constants::radii[mLayer] + constants::thickness / 2.};
};
} // namespace its3
} // namespace o2

#endif
