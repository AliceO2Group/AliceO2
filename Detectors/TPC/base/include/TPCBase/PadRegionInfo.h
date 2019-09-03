// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   PadRegionInfo.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Information storage class for the different pad regions
///
/// This class stores information of the different pad pad regions
/// like hight and with of the pads, the global row number offset,
/// number of pad rows, number of pads, ...
/// @see TPCBase/Mapper.h
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_PadRegionInfo_H
#define AliceO2_TPC_PadRegionInfo_H

#include <vector>

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/PadPos.h"

namespace o2
{
namespace tpc
{

class PadRegionInfo
{
 public:
  /// default constructor
  PadRegionInfo() = default;

  /// constructor
  /// @param region region number
  /// @param partition partition number (CRU in sector)
  /// @param numberOfPadRows number of pad row in this region
  /// @param padHeight height of the pads in tihs region
  /// @param padWidth width of the pads in this region
  /// @param radiusFirstRow radial position of the center of the first row in this region
  /// @param rowOffset row offset in region with same height
  /// @param xhelper helper value to calculate pads per row
  /// @param globalRowOffset global row offset in the sector
  PadRegionInfo(const unsigned char region,
                const unsigned char partition,
                const unsigned char numberOfPadRows,
                const float padHeight,
                const float padWidth,
                const float radiusFirstRow,
                const unsigned char rowOffset,
                const float xhelper,
                const unsigned char globalRowOffset);

  /// Return the pad region
  /// \return pad region
  unsigned char getRegion() const { return mRegion; }

  /// Return the partition
  /// \return partition
  unsigned char getPartition() const { return mPartition; }

  /// Return the number of pad rows in this region
  /// \return number of pad rows in this region
  unsigned char getNumberOfPadRows() const { return mNumberOfPadRows; }

  /// Return the total number of pads in this region
  /// \return total number of pads in this region
  unsigned short getNumberOfPads() const { return mNumberOfPads; }

  /// Return the pad height in this region
  /// \return pad height in this region
  float getPadHeight() const { return mPadHeight; }

  /// Return the pad width in this region
  /// \return pad width in this region
  float getPadWidth() const { return mPadWidth; }

  /// Return the radius of the first row in this region
  /// \return radius of the first row in this region
  float getRadiusFirstRow() const { return mRadiusFirstRow; }

  /// Return the row offset in the sector
  /// \return row offset in the sector
  unsigned char getGlobalRowOffset() const { return mGlobalRowOffset; }
  //   const unsigned char  getRowOffset()        const { return mRowOffset;        }
  //   const float          getXhelper()         const { return mXhelper;         }

  /// Return the number of pads for the row in `padPos` (row in the sector)
  /// \param padPos pad position in the sector
  /// \return number of pads for the row in `padPos` (row in the sector)
  unsigned char getPadsInRow(const PadPos& padPos) const
  {
    return mPadsPerRow[padPos.getRow() - mGlobalRowOffset];
  }

  /// Return the number of pads for the `row` (row in the sector)
  /// \param padPos row in the sector
  /// \return number of pads for the row in in the sector
  unsigned char getPadsInRow(const int row) const { return mPadsPerRow[row - mGlobalRowOffset]; }

  /// Return the number of pads for the `row` (row in the pad region)
  /// \param padPos row in the pad region
  /// \return number of pads for the row in in the pad region
  unsigned char getPadsInRowRegion(const int row) const { return mPadsPerRow[row]; }

  /// Check if the local X position is in the region
  /// \param localX local X position
  /// \param border an optional border width
  /// \return true if the local X position is in the region
  /// \todo check function
  bool isInRegion(float localX, float border = 0.f) const
  {
    return localX - mRadiusFirstRow - border > 0.f && localX - mRadiusFirstRow < (mNumberOfPadRows + 1) * mPadHeight + border;
  }

  /// Find the pad and row for a local 3D position
  /// \param pos 3D position in local coordinates
  /// \return pad and row for a local 3D position
  const PadPos findPad(const LocalPosition3D& pos) const;

  /// Find the pad and row for a local 2D position and readout side
  /// \param pos 2D position in local coordinates
  /// \param side readout side
  /// \return pad and row for a local 2D position and return side
  const PadPos findPad(const LocalPosition2D& pos, const Side side = Side::A) const;

  /// Find the pad and row for a local X and Y position and readout side
  /// \param localX local X position in local coordinates
  /// \param localY local Y position in local coordinates
  /// \param side readout side
  /// \return pad and row for a local X and Y position and readout side
  const PadPos findPad(const float localX, const float localY, const Side side = Side::A) const;

 private:
  float mPadHeight{0.f};             ///< pad height in this region
  float mPadWidth{0.f};              ///< pad width in this region
  float mRadiusFirstRow{0.f};        ///< radial position of first row
  float mXhelper{0.f};               ///< helper value to calculate pads per row
  unsigned short mNumberOfPads{0};   ///< total number of pads in region
  unsigned char mPartition{0};       ///< partition number
  unsigned char mRegion{0};          ///< pad region number
  unsigned char mNumberOfPadRows{0}; ///< number of rows in region

  unsigned char mRowOffset{0}; ///< row offset in region with same height

  unsigned char mGlobalRowOffset{0}; ///< global pad row offset

  void init();

  std::vector<unsigned char> mPadsPerRow; ///< number of pad in each row
};

} // namespace tpc
} // namespace o2
#endif
