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

/// \file Segmentation.h
/// \brief Definition of the Segmentation class
/// \author Giorgio Alberto Lucia: giorgio.alberto.lucia@cern.ch

#ifndef ALICEO2_IOTOF_SEGMENTATION_H
#define ALICEO2_IOTOF_SEGMENTATION_H

#include <Rtypes.h>
#include "MathUtils/Cartesian.h"

namespace o2
{
namespace iotof
{

/// Segmentation and response for pixels in inner and outer TOF of the ALICE 3 apparatus
/// Questions to solve:
class Segmentation
{
 public:
  
  static int NCols;
  static int NRows;
  static int NPixels;
  static float PitchCol;
  static float PitchRow;
  static float PassiveEdgeReadOut;
  static float PassiveEdgeTop;
  static float PassiveEdgeSide;
  static float ActiveMatrixSizeCols;
  static float ActiveMatrixSizeRows;

  // effective thickness of sensitive layer, accounting for charge collection non-unifoemity, https://alice.its.cern.ch/jira/browse/AOC-46
  static float SensorLayerThicknessEff;
  static float SensorLayerThickness;
  static float SensorSizeCols;
  static float SensorSizeRows;

  Segmentation();
  ~Segmentation() = default;

  static void configChip(const int nCols, const int nRows, const float pitchCol, const float pitchRow, const float passiveEdgeReadOut, const float passiveEdgeTop,
                  const float passiveEdgeSide, const float sensorLayerThicknessEff, const float sensorLayerThickness);

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

  static float getFirstRowCoordinate()
  {
    return 0.5 * ((ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - PitchRow);
  }
  static float getFirstColCoordinate() { return 0.5 * (PitchCol - ActiveMatrixSizeCols); }

  static void print();

  ClassDefNV(Segmentation, 1); // Segmentation class upgrade pixels
};

//_________________________________________________________________________________________________
inline void Segmentation::localToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol)
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
inline bool Segmentation::localToDetector(float xRow, float zCol, int& iRow, int& iCol)
{
  // convert to row/col
  xRow = 0.5 * (ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - xRow; // coordinate wrt top edge of Active matrix
  zCol += 0.5 * ActiveMatrixSizeCols;                                               // coordinate wrt left edge of Active matrix
  if (xRow < 0 || xRow >= ActiveMatrixSizeRows || zCol < 0 || zCol >= ActiveMatrixSizeCols) {
    iRow = iCol = -1;
    return false;
  }
  iRow = int(xRow / PitchRow);
  iCol = int(zCol / PitchCol);
  return true;
}

} // namespace iotof
} // namespace o2

#endif