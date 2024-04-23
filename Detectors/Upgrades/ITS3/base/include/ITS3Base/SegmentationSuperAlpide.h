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
/// \author felix.schlepper@cern.ch

#ifndef ALICEO2_ITS3_SEGMENTATIONSUPERALPIDE_H_
#define ALICEO2_ITS3_SEGMENTATIONSUPERALPIDE_H_

#include "MathUtils/Cartesian.h"
#include "ITS3Base/SpecsV2.h"
#include "Rtypes.h"

#include <type_traits>

namespace o2::its3
{

/// Segmentation and response for pixels in ITS3 upgrade
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
  virtual ~SegmentationSuperAlpide() = default;
  SegmentationSuperAlpide(const SegmentationSuperAlpide&) = default;
  SegmentationSuperAlpide(SegmentationSuperAlpide&&) = delete;
  SegmentationSuperAlpide& operator=(const SegmentationSuperAlpide&) = delete;
  SegmentationSuperAlpide& operator=(SegmentationSuperAlpide&&) = delete;
  constexpr SegmentationSuperAlpide(int layer) : mLayer{layer} {}

  static constexpr int mNCols{constants::pixelarray::nCols};
  static constexpr int mNRows{constants::pixelarray::nRows};
  static constexpr int nPixels{mNCols * mNRows};
  static constexpr float mLength{constants::pixelarray::length};
  static constexpr float mWidth{constants::pixelarray::width};
  static constexpr float mPitchCol{constants::pixelarray::length / static_cast<float>(mNCols)};
  static constexpr float mPitchRow{constants::pixelarray::width / static_cast<float>(mNRows)};
  static constexpr float mSensorLayerThickness{constants::thickness};
  static constexpr float mSensorLayerThicknessEff{constants::effThickness};
  static constexpr std::array<float, constants::nLayers> mRadii{constants::radii};
  static constexpr std::array<float, constants::nLayers> mEffRadii{mRadii[0] + constants::thickness / 2.,
                                                                   mRadii[1] + constants::thickness / 2.,
                                                                   mRadii[2] + constants::thickness / 2.};

  /// Transformation from the curved surface to a flat surface
  /// \param xCurved Detector local curved coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yCurved Detector local curved coordinate y in cm with respect to
  /// the center of the sensitive volume.
  /// \param xFlat Detector local flat coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yFlat Detector local flat coordinate y in cm with respect to
  /// the center of the sensitive volume.
  void curvedToFlat(const float xCurved, const float yCurved, float& xFlat, float& yFlat) const noexcept
  {
    // MUST align the flat surface with the curved surface with the original pixel array is on
    float dist = std::hypot(xCurved, yCurved);
    float phiReadout = constants::tile::readout::width / constants::radii[mLayer];
    float phi = std::atan2(yCurved, xCurved);
    xFlat = mRadii[mLayer] * (phi - phiReadout) - constants::pixelarray::width / 2.;
    yFlat = dist - mRadii[mLayer];
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
  void flatToCurved(float xFlat, float yFlat, float& xCurved, float& yCurved) const noexcept
  {
    // MUST align the flat surface with the curved surface with the original pixel array is on
    float dist = yFlat + mRadii[mLayer];
    float phiReadout = constants::tile::readout::width / mRadii[mLayer];
    xCurved = dist * std::cos(phiReadout + (xFlat + constants::pixelarray::width / 2.) / mRadii[mLayer]);
    yCurved = dist * std::sin(phiReadout + (xFlat + constants::pixelarray::width / 2.) / mRadii[mLayer]);
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
  bool localToDetector(float const xRow, float const zCol, int& iRow, int& iCol) const noexcept
  {
    localToDetectorUnchecked(xRow, zCol, iRow, iCol);
    if (!isValid(iRow, iCol)) {
      iRow = iCol = -1;
      return false;
    }
    return true;
  }

  // Same as localToDetector w.o. checks.
  void localToDetectorUnchecked(float const xRow, float const zCol, int& iRow, int& iCol) const noexcept
  {
    namespace cp = constants::pixelarray;
    iRow = std::floor((cp::width / 2. - xRow) / mPitchRow);
    iCol = std::floor((zCol + cp::length / 2.) / mPitchCol);
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
  bool detectorToLocal(int const iRow, int const iCol, float& xRow, float& zCol) const noexcept
  {
    if (!isValid(iRow, iCol)) {
      return false;
    }
    detectorToLocalUnchecked(iRow, iCol, xRow, zCol);
    return isValid(xRow, zCol);
  }

  // Same as detectorToLocal w.o. checks.
  // We position ourself in the middle of the pixel.
  void detectorToLocalUnchecked(int const iRow, int const iCol, float& xRow, float& zCol) const noexcept
  {
    namespace cp = constants::pixelarray;
    xRow = -(iRow + 0.5) * mPitchRow + cp::width / 2.;
    zCol = (iCol + 0.5) * mPitchCol - cp::length / 2.;
  }

  bool detectorToLocal(int const row, int const col, math_utils::Point3D<float>& loc) const noexcept
  {
    float xRow{0.}, zCol{0.};
    if (!detectorToLocal(row, col, xRow, zCol)) {
      return false;
    }
    loc.SetCoordinates(xRow, 0., zCol);
    return true;
  }

  void detectorToLocalUnchecked(int const row, int const col, math_utils::Point3D<float>& loc) const noexcept
  {
    float xRow{0.}, zCol{0.};
    detectorToLocalUnchecked(row, col, xRow, zCol);
    loc.SetCoordinates(xRow, 0., zCol);
  }

 private:
  template <typename T>
  [[nodiscard]] bool isValid(T const row, T const col) const noexcept
  {
    if constexpr (std::is_floating_point_v<T>) { // compares in local coord.
      namespace cp = constants::pixelarray;
      return !static_cast<bool>(row <= -cp::width / 2. || cp::width / 2. <= row || col <= -cp::length / 2. || cp::length / 2. <= col);
    } else { // compares in rows/cols
      return !static_cast<bool>(row < 0 || row >= static_cast<int>(mNRows) || col < 0 || col >= static_cast<int>(mNCols));
    }
  }

  const int mLayer{0}; ///< chip layer

  ClassDef(SegmentationSuperAlpide, 1);
};

/// Segmentation array
extern const std::array<SegmentationSuperAlpide, constants::nLayers> SuperSegmentations;
} // namespace o2::its3

#endif
