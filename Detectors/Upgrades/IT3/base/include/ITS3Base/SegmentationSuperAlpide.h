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

#ifndef ALICEO2_ITS3_SEGMENTATIONSUPERALPIDE_H_
#define ALICEO2_ITS3_SEGMENTATIONSUPERALPIDE_H_

#include <Rtypes.h>
#include "MathUtils/Cartesian.h"
#include "CommonConstants/MathConstants.h"

namespace o2
{
namespace its3
{

/// Segmentation and response for pixels in ITS3 upgrade
class SegmentationSuperAlpide
{
 public:
  SegmentationSuperAlpide(int layer = 0) : mLayer{layer},
                                           NPixels{NRows * NCols},
                                           NRows{static_cast<int>(double(Radii[layer]) * double(constants::math::TwoPI) / double(PitchRow) + 1)},
                                           ActiveMatrixSizeRows{PitchRow * NRows},
                                           SensorSizeRows{ActiveMatrixSizeRows + PassiveEdgeTop + PassiveEdgeReadOut}
  {
  }
  int mLayer;
  static constexpr int NLayers = 4;
  static constexpr float Length = 27.15f;
  static constexpr float Radii[NLayers] = {1.8f, 2.4f, 3.0f, 7.0f};
  static constexpr float PitchCol = 29.24e-4;
  static constexpr float PitchRow = 26.88e-4;
  static constexpr int NCols = Length / PitchCol;
  int NRows;
  int NPixels;
  static constexpr float PassiveEdgeReadOut = 0.;                 // width of the readout edge (Passive bottom)
  static constexpr float PassiveEdgeTop = 0.;                     // Passive area on top
  static constexpr float PassiveEdgeSide = 0.;                    // width of Passive area on left/right of the sensor
  static constexpr float ActiveMatrixSizeCols = PitchCol * NCols; // Active size along columns
  float ActiveMatrixSizeRows;                                     // Active size along rows

  // effective thickness of sensitive layer, accounting for charge collection non-unifoemity, https://alice.its.cern.ch/jira/browse/AOC-46
  static constexpr float SensorLayerThicknessEff = 28.e-4;
  static constexpr float SensorLayerThickness = 30.e-4;                                             // physical thickness of sensitive part
  static constexpr float SensorSizeCols = ActiveMatrixSizeCols + PassiveEdgeSide + PassiveEdgeSide; // SensorSize along columns
  float SensorSizeRows;                                                                             // SensorSize along rows

  ~SegmentationSuperAlpide() = default;

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
  bool localToDetector(float x, float z, int& iRow, int& iCol);
  /// same but w/o check for row/column range
  void localToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol);

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
  bool detectorToLocal(int iRow, int iCol, float& xRow, float& zCol);
  bool detectorToLocal(float row, float col, float& xRow, float& zCol);
  bool detectorToLocal(float row, float col, math_utils::Point3D<float>& loc);

  // same but w/o check for row/col range
  void detectorToLocalUnchecked(int iRow, int iCol, float& xRow, float& zCol);
  void detectorToLocalUnchecked(float row, float col, float& xRow, float& zCol);
  void detectorToLocalUnchecked(float row, float col, math_utils::Point3D<float>& loc);

  float getFirstRowCoordinate()
  {
    return 0.5 * ((ActiveMatrixSizeRows - PassiveEdgeTop + PassiveEdgeReadOut) - PitchRow);
  }
  static constexpr float getFirstColCoordinate() { return 0.5 * (PitchCol - ActiveMatrixSizeCols); }

  void print();

  ClassDefNV(SegmentationSuperAlpide, 1); // Segmentation class upgrade pixels
};

inline void SegmentationSuperAlpide::localToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol)
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

inline bool SegmentationSuperAlpide::localToDetector(float xRow, float zCol, int& iRow, int& iCol)
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

inline void SegmentationSuperAlpide::detectorToLocalUnchecked(int iRow, int iCol, float& xRow, float& zCol)
{
  xRow = getFirstRowCoordinate() - iRow * PitchRow;
  zCol = iCol * PitchCol + getFirstColCoordinate();
}

inline void SegmentationSuperAlpide::detectorToLocalUnchecked(float row, float col, float& xRow, float& zCol)
{
  xRow = getFirstRowCoordinate() - row * PitchRow;
  zCol = col * PitchCol + getFirstColCoordinate();
}

inline void SegmentationSuperAlpide::detectorToLocalUnchecked(float row, float col, math_utils::Point3D<float>& loc)
{
  loc.SetCoordinates(getFirstRowCoordinate() - row * PitchRow, 0.f, col * PitchCol + getFirstColCoordinate());
}

inline bool SegmentationSuperAlpide::detectorToLocal(int iRow, int iCol, float& xRow, float& zCol)
{
  if (iRow < 0 || iRow >= NRows || iCol < 0 || iCol >= NCols) {
    return false;
  }
  detectorToLocalUnchecked(iRow, iCol, xRow, zCol);
  return true;
}

inline bool SegmentationSuperAlpide::detectorToLocal(float row, float col, float& xRow, float& zCol)
{
  if (row < 0 || row >= NRows || col < 0 || col >= NCols) {
    return false;
  }
  detectorToLocalUnchecked(row, col, xRow, zCol);
  return true;
}

inline bool SegmentationSuperAlpide::detectorToLocal(float row, float col, math_utils::Point3D<float>& loc)
{
  if (row < 0 || row >= NRows || col < 0 || col >= NCols) {
    return false;
  }
  detectorToLocalUnchecked(row, col, loc);
  return true;
}
} // namespace its3
} // namespace o2

#endif
