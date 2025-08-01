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

/// \file SegmentationChip.h
/// \brief Definition of the SegmentationChipclass

#ifndef ALICEO2_TRK_SEGMENTATIONCHIP_H_
#define ALICEO2_TRK_SEGMENTATIONCHIP_H_

#include <type_traits>
#include <fairlogger/Logger.h>

#include "MathUtils/Cartesian.h"
#include "TRKBase/Specs.h"

namespace o2::trk
{

/// Segmentation and response for TRK chips in ALICE3 upgrade
/// This is a work-in-progress code derived from the ITS2 and ITS3 segmentations.
class SegmentationChip
{
  // This class defines the segmenation of the TRK chips in the ALICE3 upgrade. We define
  // two coordinate systems, one width x,z detector local coordianates (cm) and
  // the more natural row,col layout: Also all the transformation between these
  // two. The class provides the transformation from the stave/layer/disk to TGeo
  // coordinates.
  // For the curved VD layers there exist three coordinate systems and one is transient.
  // 1. The global (curved) coordinate system. The chip's center of coordinate system is
  //    defined at the the mid-point of the detector.
  // 2. The flat coordinate system. This is the tube segment projected onto a flat
  //    surface. In the projection we implicitly assume that the inner and outer
  //    stretch does not depend on the radius.
  // 3. The detector coordinate system. Defined by the row and column segmentation
  //    defined at the upper edge in the flat coord.
  // For the flat ML and OT layers, there exist two coordinate systems:
  // 1. The global (flat) coordinate system. The chip's center of coordinate system is
  //    defined at the the mid-point of the detector.
  // 2. The detector coordinate system. Defined by the row and column segmentation
  // TODO: add segmentation for VD disks

 public:
  constexpr SegmentationChip() = default;
  ~SegmentationChip() = default;
  constexpr SegmentationChip(const SegmentationChip&) = default;
  constexpr SegmentationChip(SegmentationChip&&) = delete;
  constexpr SegmentationChip& operator=(const SegmentationChip&) = default;
  constexpr SegmentationChip& operator=(SegmentationChip&&) = delete;

  static constexpr float PitchColVD{constants::VD::petal::layer::pitchZ};
  static constexpr float PitchRowVD{constants::VD::petal::layer::pitchX};

  static constexpr float PitchColML{constants::moduleMLOT::chip::pitchZ};
  static constexpr float PitchRowML{constants::moduleMLOT::chip::pitchX};

  static constexpr float PitchColOT{constants::moduleMLOT::chip::pitchZ};
  static constexpr float PitchRowOT{constants::moduleMLOT::chip::pitchX};

  static constexpr float SensorLayerThicknessVD = {constants::VD::petal::layer::totalThickness}; // physical thickness of sensitive part = 30 um
  static constexpr float SensorLayerThicknessML = {constants::moduleMLOT::chip::totalThickness}; // physical thickness of sensitive part = 100 um
  static constexpr float SensorLayerThicknessOT = {constants::moduleMLOT::chip::totalThickness}; // physical thickness of sensitive part = 100 um

  static constexpr float SiliconTickness = constants::silicon::thickness; // effective thickness of sensitive part

  static constexpr std::array<double, constants::VD::petal::nLayers> radiiVD = constants::VD::petal::layer::radii;

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
  /// \param int subDetID Sub-detector ID (0 for VD, 1 for ML/OT)
  /// \param int layer Layer number (0 to 2 for VD, 0 to 7 for ML/OT)
  /// \param int disk Disk number (0 to 5 for VD)
  static bool globalToDetector(float xRow, float zCol, int& iRow, int& iCol, int subDetID, int layer, int disk) noexcept
  {
    if (!isValidGlob(xRow, zCol, subDetID, layer)) {
      LOGP(info, "Global coordinates not valid: row = {} cm, col = {} cm", xRow, zCol);
      return false;
    }

    globalToDetectorUnchecked(xRow, zCol, iRow, iCol, subDetID, layer, disk);

    if (!isValidDet(iRow, iCol, subDetID, layer)) {
      iRow = iCol = -1;
      LOGP(info, "Detector coordinates not valid: iRow = {}, iCol = {}", iRow, iCol);
      return false;
    }
    return true;
  };
  /// same but w/o check for row/column range
  static void globalToDetectorUnchecked(float xRow, float zCol, int& iRow, int& iCol, int subDetID, int layer, int disk) noexcept
  {
    // convert to row/col w/o over/underflow check
    float pitchRow(0), pitchCol(0);
    float maxWidth(0), maxLength(0);

    if (subDetID == 0) {
      pitchRow = PitchRowVD;
      pitchCol = PitchColVD;
      maxWidth = constants::VD::petal::layer::width[layer];
      maxLength = constants::VD::petal::layer::length;
      // TODO: change this to use the layer and disk
    } else if (subDetID == 1 && layer <= 3) { // ML
      pitchRow = PitchRowML;
      pitchCol = PitchColML;
      maxWidth = constants::ML::width;
      maxLength = constants::ML::length;
    } else if (subDetID == 1 && layer >= 4) { // OT
      pitchRow = PitchRowOT;
      pitchCol = PitchColOT;
      maxWidth = constants::OT::width;
      maxLength = constants::OT::length;
    }
    // convert to row/col
    iRow = static_cast<int>(std::floor((maxWidth / 2 - xRow) / pitchRow));
    iCol = static_cast<int>(std::floor((zCol + maxLength / 2) / pitchCol));
  };

  // Check global coordinates (cm) validity.
  static constexpr bool isValidGlob(float x, float z, int subDetID, int layer) noexcept
  {
    float maxWidth(0), maxLength(0);
    if (subDetID == 0) {
      maxWidth = constants::VD::petal::layer::width[layer];
      maxLength = constants::VD::petal::layer::length;
      // TODO: change this to use the layer and disk
    } else if (subDetID == 1 && layer <= 3) { // ML
      maxWidth = constants::ML::width;
      maxLength = constants::ML::length;
    } else if (subDetID == 1 && layer >= 4) { // OT
      maxWidth = constants::OT::width;
      maxLength = constants::OT::length;
    }
    return (-maxWidth / 2 < x && x < maxWidth / 2 && -maxLength / 2 < z && z < maxLength / 2);
  }

  // Check detector coordinates validity.
  static constexpr bool isValidDet(float row, float col, int subDetID, int layer) noexcept
  {
    // Check if the row and column are within the valid range
    int nRows(0), nCols(0);
    if (subDetID == 0) {
      nRows = constants::VD::petal::layer::nRows[layer];
      nCols = constants::VD::petal::layer::nCols;
      // TODO: change this to use the layer and disk
    } else if (subDetID == 1 && layer <= 3) { // ML
      nRows = constants::ML::nRows;
      nCols = constants::ML::nCols;
    } else if (subDetID == 1 && layer >= 4) { // OT
      nRows = constants::OT::nRows;
      nCols = constants::OT::nCols;
    }
    return (row >= 0 && row < static_cast<float>(nRows) &&
            col >= 0 && col < static_cast<float>(nCols));
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
  /// \param int subDetID Sub-detector ID (0 for VD, 1 for ML/OT)
  /// \param int layer Layer number (0 to 2 for VD, 0 to 7 for ML/OT)
  /// \param int disk Disk number (0 to 5 for VD)
  static constexpr bool detectorToGlobal(int iRow, int iCol, float& xRow, float& zCol, int subDetID, int layer, int disk) noexcept
  {
    if (!isValidDet(xRow, zCol, subDetID, layer)) {
      LOGP(debug, "Detector coordinates not valid: iRow = {}, iCol = {}", iRow, iCol);
      return false;
    }
    detectorToGlobalUnchecked(iRow, iCol, xRow, zCol, subDetID, layer, disk);

    if (!isValidGlob(xRow, zCol, subDetID, layer)) {
      LOGP(debug, "Global coordinates not valid: row = {} cm, col = {} cm", xRow, zCol);
      return false;
    }
    return true;
  };

  // Same as detectorToGlobal w.o. checks.
  // We position ourself in the middle of the pixel.
  static void detectorToGlobalUnchecked(int& row, int& col, float xRow, float zCol, int subDetID, int layer, int disk) noexcept
  {
    if (subDetID == 0) {
      xRow = -(row + 0.5f) * PitchRowVD + constants::VD::petal::layer::width[layer] / 2;
      zCol = (col + 0.5f) * PitchColVD - constants::VD::petal::layer::length / 2;
    } else if (subDetID == 1 && layer <= 3) { // ML
      xRow = -(row + 0.5f) * PitchRowML + constants::ML::width / 2;
      zCol = (col + 0.5f) * PitchColML - constants::ML::length / 2;
    } else if (subDetID == 1 && layer >= 4) { // OT
      xRow = -(row + 0.5f) * PitchRowOT + constants::OT::width / 2;
      zCol = (col + 0.5f) * PitchColOT - constants::OT::length / 2;
    }
  }

  /// Transformation from the curved surface to a flat surface.
  /// Additionally a shift in the flat coordinates must be applied because
  /// the center of the TGeoShap when projected will be higher than the
  /// physical thickness of the chip. Thus we shift the projected center
  /// down by this difference to align the coordinate systems.
  /// \param layer VD layer number
  /// \param xCurved Detector local curved coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yCurved Detector local curved coordinate y in cm with respect to
  /// the center of the sensitive volume.
  /// \return math_utils::Vector2D<float>: x and y represent the detector local flat coordinates x and y
  // in cm with respect to the center of the sensitive volume.

  static math_utils::Vector2D<float> curvedToFlat(const int layer, const float xCurved, const float yCurved) noexcept
  {
    // Align the flat surface with the curved survace of the original chip (and account for metal stack, TODO)
    float dist = std::hypot(xCurved, yCurved);
    float phi = std::atan2(yCurved, xCurved);

    // the y position is in the silicon volume however we need the chip volume (silicon+metalstack)
    // this is accounted by a y shift
    float xFlat = -constants::VD::petal::layer::radii[layer] * phi; /// this is equal to the circumference segment covered between y=0 and the phi angle
    float yFlat = dist - constants::VD::petal::layer::radii[layer];
    return math_utils::Vector2D<float>(xFlat, yFlat);
  }

  /// Transformation from the flat surface to a curved surface
  /// It works only if the detector is not rototraslated.
  /// \param layer VD layer number
  /// \param xFlat Detector local flat coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param yFlat Detector local flat coordinate y in cm with respect to
  /// the center of the sensitive volume.
  /// \return math_utils::Vector2D<float>: x and y represent the detector local curved coordinates x and y
  // in cm with respect to the center of the sensitive volume.
  static constexpr math_utils::Vector2D<float> flatToCurved(int layer, float xFlat, float yFlat) noexcept
  {
    // Revert the curvedToFlat transformation
    float dist = yFlat + constants::VD::petal::layer::radii[layer];
    float phi = -xFlat / constants::VD::petal::layer::radii[layer];
    // the y position is in the chip volume however we need the silicon volume
    // this is accounted by a -y shift
    float xCurved = dist * std::cos(phi);
    float yCurved = dist * std::sin(phi);
    return math_utils::Vector2D<float>(xCurved, yCurved);
  }
};

} // namespace o2::trk

#endif