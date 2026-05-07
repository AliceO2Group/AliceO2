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
#include <memory>
#include "MathUtils/Cartesian.h"
#include "IOTOFBase/IOTOFBaseParam.h"

namespace o2
{
namespace iotof
{

/// Segmentation and response for pixels in inner and outer TOF of the ALICE 3 apparatus
/// Questions to solve:
class Segmentation
{
 private:
  Segmentation();
  static std::unique_ptr<o2::iotof::Segmentation> sInstance;

 public:
  ChipSpecifics mITofSpecsConfig;
  ChipSpecifics mOTofSpecsConfig;
  static Segmentation* Instance();

  ~Segmentation() = default;

  void configChip(const int nCols, const int nRows, const float pitchCol, const float pitchRow, const float passiveEdgeReadOut, const float passiveEdgeTop,
                  const float passiveEdgeSide, const float sensorLayerThicknessEff, const float sensorLayerThickness, const int subDetectorID);
  void configChip(const ChipSpecifics& specsConfig, const int subDetectorID);

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
  bool localToDetector(float x, float z, int& iRow, int& iCol, const int subDetectorID);
  /// same but w/o check for row/column range
  void localToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol, const int subDetectorID);

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
  void detectorToLocalUnchecked(L row, L col, T& xRow, T& zCol, const int subDetectorID)
  {
    if (subDetectorID != 0 && subDetectorID != 1) {
      row = col = -1;
      return;
    }
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    xRow = getFirstRowCoordinate(subDetectorID) - row * specsConfig.PitchRow;
    zCol = col * specsConfig.PitchCol + getFirstColCoordinate(subDetectorID);
  }
  template <typename T = float, typename L = float>
  void detectorToLocalUnchecked(L row, L col, math_utils::Point3D<T>& loc, const int subDetectorID)
  {
    if (subDetectorID != 0 && subDetectorID != 1) {
      row = col = -1;
      return;
    }
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    loc.SetCoordinates(getFirstRowCoordinate(subDetectorID) - row * specsConfig.PitchRow, T(0.), col * specsConfig.PitchCol + getFirstColCoordinate(subDetectorID));
  }
  template <typename T = float, typename L = float>
  void detectorToLocalUnchecked(L row, L col, std::array<T, 3>& loc, const int subDetectorID)
  {
    if (subDetectorID != 0 && subDetectorID != 1) {
      row = col = -1;
      return;
    }
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    loc[0] = getFirstRowCoordinate(subDetectorID) - row * specsConfig.PitchRow;
    loc[1] = T(0);
    loc[2] = col * specsConfig.PitchCol + getFirstColCoordinate(subDetectorID);
  }

  // same but with check for row/col range

  template <typename T = float, typename L = float>
  bool detectorToLocal(L row, L col, T& xRow, T& zCol, const int subDetectorID)
  {
    if (subDetectorID != 0 && subDetectorID != 1) {
      row = col = -1;
      return false;
    }
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    if (row < 0 || row >= specsConfig.NRows || col < 0 || col >= specsConfig.NCols) {
      return false;
    }
    detectorToLocalUnchecked(row, col, xRow, zCol, subDetectorID);
    return true;
  }

  template <typename T = float, typename L = float>
  bool detectorToLocal(L row, L col, math_utils::Point3D<T>& loc, const int subDetectorID)
  {
    if (subDetectorID != 0 && subDetectorID != 1) {
      row = col = -1;
      return false;
    }
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    if (row < 0 || row >= specsConfig.NRows || col < 0 || col >= specsConfig.NCols) {
      return false;
    }
    detectorToLocalUnchecked(row, col, loc, subDetectorID);
    return true;
  }
  template <typename T = float, typename L = float>
  bool detectorToLocal(L row, L col, std::array<T, 3>& loc, const int subDetectorID)
  {
    if (subDetectorID != 0 && subDetectorID != 1) {
      row = col = -1;
      return false;
    }
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    if (row < 0 || row >= specsConfig.NRows || col < 0 || col >= specsConfig.NCols) {
      return false;
    }
    detectorToLocalUnchecked(row, col, loc, subDetectorID);
    return true;
  }

  float getFirstRowCoordinate(const int subDetectorID)
  {
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    return 0.5 * ((specsConfig.ActiveMatrixSizeRows() - specsConfig.PassiveEdgeTop + specsConfig.PassiveEdgeReadOut) - specsConfig.PitchRow);
  }
  float getFirstColCoordinate(const int subDetectorID)
  {
    const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
    return 0.5 * (specsConfig.PitchCol - specsConfig.ActiveMatrixSizeCols());
  }

  void print();

  ClassDefNV(Segmentation, 1); // Segmentation class upgrade pixels
};

//_________________________________________________________________________________________________
inline void Segmentation::localToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol, const int subDetectorID)
{
  // convert to row/col w/o over/underflow check
  if (subDetectorID != 0 && subDetectorID != 1) {
    iRow = iCol = -1;
    return;
  }
  const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
  xRow = 0.5 * (specsConfig.ActiveMatrixSizeRows() - specsConfig.PassiveEdgeTop + specsConfig.PassiveEdgeReadOut) - xRow; // coordinate wrt top edge of Active matrix
  zCol += 0.5 * specsConfig.ActiveMatrixSizeCols();                                                                       // coordinate wrt left edge of Active matrix
  iRow = int(xRow / specsConfig.PitchRow);
  iCol = int(zCol / specsConfig.PitchCol);
  if (xRow < 0) {
    iRow -= 1;
  }
  if (zCol < 0) {
    iCol -= 1;
  }
}

//_________________________________________________________________________________________________
inline bool Segmentation::localToDetector(float xRow, float zCol, int& iRow, int& iCol, const int subDetectorID)
{
  // convert to row/col
  if (subDetectorID != 0 && subDetectorID != 1) {
    iRow = iCol = -1;
    return false;
  }
  const ChipSpecifics& specsConfig = (subDetectorID == 0) ? mITofSpecsConfig : mOTofSpecsConfig;
  xRow = 0.5 * (specsConfig.ActiveMatrixSizeRows() - specsConfig.PassiveEdgeTop + specsConfig.PassiveEdgeReadOut) - xRow; // coordinate wrt top edge of Active matrix
  zCol += 0.5 * specsConfig.ActiveMatrixSizeCols();                                                                       // coordinate wrt left edge of Active matrix
  if (xRow < 0 || xRow >= specsConfig.ActiveMatrixSizeRows() || zCol < 0 || zCol >= specsConfig.ActiveMatrixSizeCols()) {
    iRow = iCol = -1;
    return false;
  }
  iRow = int(xRow / specsConfig.PitchRow);
  iCol = int(zCol / specsConfig.PitchCol);
  return true;
}

} // namespace iotof
} // namespace o2

#endif
