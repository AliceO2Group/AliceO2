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

/// \file SegmentationMosaix.h
/// \brief Definition of the SegmentationMosaix class
/// \author felix.schlepper@cern.ch
/// \author chunzheng.wang@cern.ch

#ifndef ALICEO2_ITS3_SEGMENTATIONMOSAIX_H_
#define ALICEO2_ITS3_SEGMENTATIONMOSAIX_H_

#include "MathUtils/Cartesian.h"
#include "ITS3Base/SpecsV2.h"

namespace o2::its3
{

/// Segmentation and response for pixels in ITS3 upgrade
class SegmentationMosaix
{
  // This class defines the segmenation of the pixelArray in the tile. We define
  // two coordinate systems, one width x,z detector local coordianates (cm) and
  // the more natural row,col layout: Also all the transformation between these
  // two. The class provides the transformation from the tile to TGeo
  // coordinates.
  // In fact there exist three coordinate systems and one is transient.
  // 1. The curved coordinate system. The chip's local coordinate system is
  //    defined with its center at the the mid-point of the tube.
  // 2. The flat coordinate system. This is the tube segment projected onto a flat
  //    surface. In the projection we implicitly assume that the inner and outer
  //    stretch does not depend on the radius.
  //    Additionally, there is a difference between the flat geometrical center
  //    and the phyiscal center defined by the metal layer.
  // 3. The detector coordinate system. Defined by the row and column segmentation
  //    defined at the upper edge in the flat coord.

  // O----------------------|
  // |           |          |
  // |           |          |  ^ x
  // |           |          |  |
  // |           |          |  |
  // |           |          |  |
  // |           |          |  X----> z   X marks (x,z)=(0,0)
  // |-----------X----------|
  // |           |          |  O----> col O marks (row,col)=(0,0)
  // |           |          |  |
  // |           |          |  |
  // |           |          |  v
  // |           |          |  row
  // |           |          |
  // |----------------------|

 public:
  constexpr SegmentationMosaix(int layer) : mRadius(static_cast<float>(constants::radiiMiddle[layer])) {}
  constexpr ~SegmentationMosaix() = default;
  constexpr SegmentationMosaix(const SegmentationMosaix&) = default;
  constexpr SegmentationMosaix(SegmentationMosaix&&) = delete;
  constexpr SegmentationMosaix& operator=(const SegmentationMosaix&) = default;
  constexpr SegmentationMosaix& operator=(SegmentationMosaix&&) = delete;

  static constexpr int NCols{constants::pixelarray::nCols};
  static constexpr int NRows{constants::pixelarray::nRows};
  static constexpr int NPixels{NCols * NRows};
  static constexpr float Length{constants::pixelarray::length};
  static constexpr float LengthH{Length / 2.f};
  static constexpr float Width{constants::pixelarray::width};
  static constexpr float WidthH{Width / 2.f};
  static constexpr float PitchCol{constants::pixelarray::pixels::mosaix::pitchZ};
  static constexpr float PitchRow{constants::pixelarray::pixels::mosaix::pitchX};
  static constexpr float SensorLayerThickness{constants::totalThickness};

  /// Transformation from the curved surface to a flat surface.
  /// Additionally a shift in the flat coordinates must be applied because
  /// the center of the TGeoShap when projected will be higher than the
  /// physical thickness of the chip (we add an additional hull to account for
  /// the copper metal interconnection which is in reality part of the chip but in our
  /// simulation the silicon and metal layer are separated). Thus we shift the projected center
  /// down by this difference to align the coordinate systems.
  /// \param xCurved Detector local curved coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yCurved Detector local curved coordinate y in cm with respect to
  /// the center of the sensitive volume.
  /// \param xFlat Detector local flat coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yFlat Detector local flat coordinate y in cm with respect to
  /// the center of the sensitive volume.
  constexpr void curvedToFlat(const float xCurved, const float yCurved, float& xFlat, float& yFlat) const noexcept
  {
    // MUST align the flat surface with the curved surface with the original pixel array is on and account for metal
    // stack
    float dist = std::hypot(xCurved, yCurved);
    float phi = std::atan2(yCurved, xCurved);
    // the y position is in the silicon volume however we need the chip volume (silicon+metalstack)
    // this is accounted by a y shift
    xFlat = WidthH - mRadius * phi;
    yFlat = dist - mRadius;
  }

  /// Transformation from the flat surface to a curved surface
  /// It works only if the detector is not rototraslated.
  /// \param xFlat Detector local flat coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yFlat Detector local flat coordinate y in cm with respect to
  /// the center of the sensitive volume.
  /// \param xCurved Detector local curved coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yCurved Detector local curved coordinate y in cm with respect to
  /// the center of the sensitive volume.
  constexpr void flatToCurved(float xFlat, float yFlat, float& xCurved, float& yCurved) const noexcept
  {
    // MUST align the flat surface with the curved surface with the original pixel array is on and account for metal
    // stack
    float dist = yFlat + mRadius;
    float phi = (WidthH - xFlat) / mRadius;
    // the y position is in the chip volume however we need the silicon volume
    // this is accounted by a -y shift
    xCurved = dist * std::cos(phi);
    yCurved = dist * std::sin(phi);
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
  constexpr bool localToDetector(float const xRow, float const zCol, int& iRow, int& iCol) const noexcept
  {
    if (!isValidLoc(xRow, zCol)) {
      return false;
    }
    localToDetectorUnchecked(xRow, zCol, iRow, iCol);
    if (!isValidDet(iRow, iCol)) {
      iRow = iCol = -1;
      return false;
    }
    return true;
  }

  // Same as localToDetector w.o. checks.
  template <typename T = float>
  constexpr void localToDetectorUnchecked(T const xRow, T const zCol, int& iRow, int& iCol) const noexcept
  {
    iRow = static_cast<int>(std::floor((WidthH - xRow) / PitchRow));
    iCol = static_cast<int>(std::floor((zCol + LengthH) / PitchCol));
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
  template <typename T = float, typename L = float>
  bool detectorToLocal(T const row, T const col, L& xRow, L& zCol) const noexcept
  {
    if (!isValidDet(row, col)) {
      return false;
    }
    detectorToLocalUnchecked(row, col, xRow, zCol);
    return isValidLoc(xRow, zCol);
  }

  // Same as detectorToLocal w.o. checks.
  // We position ourself in the middle of the pixel.
  template <typename T = float, typename L = float>
  void detectorToLocalUnchecked(T const row, T const col, L& xRow, L& zCol) const noexcept
  {
    xRow = -(row + 0.5f) * PitchRow + WidthH;
    zCol = (col + 0.5f) * PitchCol - LengthH;
  }

  template <typename T = float, typename L = float>
  bool detectorToLocal(T const row, T const col, math_utils::Point3D<L>& loc) const noexcept
  {
    L xRow{0.}, zCol{0.};
    if (!detectorToLocal(row, col, xRow, zCol)) {
      return false;
    }
    loc.SetCoordinates(xRow, 0.0f, zCol);
    return true;
  }

  template <typename T = float, typename L = float>
  void detectorToLocalUnchecked(T const row, T const col, math_utils::Point3D<L>& loc) const noexcept
  {
    L xRow{0.}, zCol{0.};
    detectorToLocalUnchecked(row, col, xRow, zCol);
    loc.SetCoordinates(xRow, 0.0f, zCol);
  }

 private:
  // Check local coordinates (cm) validity.
  template <typename T>
  constexpr bool isValidLoc(T const x, T const z) const noexcept
  {
    return (-WidthH < x && x < WidthH && -LengthH < z && z < LengthH);
  }

  // Check detector coordinates validity.
  template <typename T>
  constexpr bool isValidDet(T const row, T const col) const noexcept
  {
    return (row >= 0 && row < static_cast<T>(NRows) &&
            col >= 0 && col < static_cast<T>(NCols));
  }

  float mRadius;
};

} // namespace o2::its3

#endif
