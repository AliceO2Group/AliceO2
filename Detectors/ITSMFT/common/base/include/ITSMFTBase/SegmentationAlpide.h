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

/// \file SegmentationAlpide.h
/// \brief Definition of the SegmentationAlpide class

#ifndef ALICEO2_ITSMFT_SEGMENTATIONALPIDE_H_
#define ALICEO2_ITSMFT_SEGMENTATIONALPIDE_H_

#include <Rtypes.h>
#include "MathUtils/Cartesian.h"

namespace o2
{
namespace itsmft
{

/// Segmentation and response for pixels in ITSMFT upgrade
/// Questions to solve: are guardrings needed and do they belong to the sensor or to the chip in
/// TGeo. At the moment assume that the local coord syst. is located at bottom left corner
/// of the ACTIVE matrix. If the guardring to be accounted in the local coords, in
/// the Z and X conversions one needs to first subtract the  mGuardLeft and mGuardBottom
/// from the local Z,X coordinates
class SegmentationAlpide
{
 public:
  static constexpr int NCols = 1024;
  static constexpr int NRows = 512;
  static constexpr int NPixels = NRows * NCols;
  static constexpr float PitchCol = 29.24e-4;
  static constexpr float PitchRow = 26.88e-4;
  static constexpr float PassiveEdgeReadOut = 0.12f;              // width of the readout edge (Passive bottom)
  static constexpr float PassiveEdgeTop = 37.44e-4;               // Passive area on top
  static constexpr float PassiveEdgeSide = 29.12e-4;              // width of Passive area on left/right of the sensor
  static constexpr float ActiveMatrixSizeCols = PitchCol * NCols; // Active size along columns
  static constexpr float ActiveMatrixSizeRows = PitchRow * NRows; // Active size along rows

  // effective thickness of sensitive layer, accounting for charge collection non-unifoemity, https://alice.its.cern.ch/jira/browse/AOC-46
  static constexpr float SensorLayerThicknessEff = 28.e-4;
  static constexpr float SensorLayerThickness = 30.e-4;                                               // physical thickness of sensitive part
  static constexpr float SensorSizeCols = ActiveMatrixSizeCols + PassiveEdgeSide + PassiveEdgeSide;   // SensorSize along columns
  static constexpr float SensorSizeRows = ActiveMatrixSizeRows + PassiveEdgeTop + PassiveEdgeReadOut; // SensorSize along rows

  SegmentationAlpide() = default;
  ~SegmentationAlpide() = default;

  /// Transformation from Geant detector centered local coordinates (cm) to
  /// Pixel cell numbers iRow and iCol.
  /// Returns kTRUE if point x,z is inside sensitive volume, kFALSE otherwise.
  /// A value of -1 for iRow or iCol indicates that this point is outside of the
  /// detector segmentation as defined.
  /// \param float x Detector local coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param float z Detector local coordinate z in cm with respect to
  /// the center of the sensitive volulme.
  /// \param int iRow Detector x cell coordinate. Has the range 0 <= iRow < mNumberOfRows
  /// \param int iCol Detector z cell coordinate. Has the range 0 <= iCol < mNumberOfColumns
  static bool localToDetector(float x, float z, int& iRow, int& iCol);
  /// same but w/o check for row/column range
  static void localToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol);

  /// Transformation from Detector cell coordiantes to Geant detector centered
  /// local coordinates (cm)
  /// \param int iRow Detector x cell coordinate. Has the range 0 <= iRow < mNumberOfRows
  /// \param int iCol Detector z cell coordinate. Has the range 0 <= iCol < mNumberOfColumns
  /// \param float x Detector local coordinate x in cm with respect to the
  /// center of the sensitive volume.
  /// \param float z Detector local coordinate z in cm with respect to the
  /// center of the sensitive volulme.
  /// If iRow and or iCol is outside of the segmentation range a value of -0.5*Dx()
  /// or -0.5*Dz() is returned.

  // w/o check for row/col range
  template <typename T = float, typename L = float>
  static void detectorToLocalUnchecked(L row, L col, T& xRow, T& zCol)
  {
    xRow = getFirstRowCoordinate() - row * PitchRow;
    zCol = col * PitchCol + getFirstColCoordinate();
  }
  template <typename T = float, typename L = float>
  static void detectorToLocalUnchecked(L row, L col, math_utils::Point3D<T>& loc)
  {
    loc.SetCoordinates(getFirstRowCoordinate() - row * PitchRow, T(0.), col * PitchCol + getFirstColCoordinate());
  }
  template <typename T = float, typename L = float>
  static void detectorToLocalUnchecked(L row, L col, std::array<T, 3>& loc)
  {
    loc[0] = getFirstRowCoordinate() - row * PitchRow;
    loc[1] = T(0);
    loc[2] = col * PitchCol + getFirstColCoordinate();
  }

  // same but with check for row/col range

  template <typename T = float, typename L = float>
  static bool detectorToLocal(L row, L col, T& xRow, T& zCol)
  {
    if (row < 0 || row >= NRows || col < 0 || col >= NCols) {
      return false;
    }
    detectorToLocalUnchecked(row, col, xRow, zCol);
    return true;
  }

  template <typename T = float, typename L = float>
  static bool detectorToLocal(L row, L col, math_utils::Point3D<T>& loc)
  {
    if (row < 0 || row >= NRows || col < 0 || col >= NCols) {
      return false;
    }
    detectorToLocalUnchecked(row, col, loc);
    return true;
  }
  template <typename T = float, typename L = float>
  static bool detectorToLocal(L row, L col, std::array<T, 3>& loc)
  {
    if (row < 0 || row >= NRows || col < 0 || col >= NCols) {
      return false;
    }
    detectorToLocalUnchecked(row, col, loc);
    return true;
  }

  static constexpr float getFirstRowCoordinate()
  {
    return 0.5 * ((ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - PitchRow);
  }
  static constexpr float getFirstColCoordinate() { return 0.5 * (PitchCol - ActiveMatrixSizeCols); }

  static void print();

  ClassDefNV(SegmentationAlpide, 1); // Segmentation class upgrade pixels
};

//_________________________________________________________________________________________________
inline void SegmentationAlpide::localToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol)
{
  // convert to row/col w/o over/underflow check
  xRow = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - xRow; // coordinate wrt top edge of Active matrix
  zCol += 0.5 * ActiveMatrixSizeCols;                                               // coordinate wrt left edge of Active matrix
  iRow = int(xRow / PitchRow);
  iCol = int(zCol / PitchCol);
  if (xRow < 0) {
    iRow -= 1;
  }
  if (zCol < 0) {
    iCol -= 1;
  }
}

//_________________________________________________________________________________________________
inline bool SegmentationAlpide::localToDetector(float xRow, float zCol, int& iRow, int& iCol)
{
  // convert to row/col
  xRow = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - xRow; // coordinate wrt left edge of Active matrix
  zCol += 0.5 * ActiveMatrixSizeCols;                                               // coordinate wrt bottom edge of Active matrix
  if (xRow < 0 || xRow >= ActiveMatrixSizeRows || zCol < 0 || zCol >= ActiveMatrixSizeCols) {
    iRow = iCol = -1;
    return false;
  }
  iRow = int(xRow / PitchRow);
  iCol = int(zCol / PitchCol);
  return true;
}

} // namespace itsmft
} // namespace o2

#endif
