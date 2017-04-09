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

#include "TPCBase/Defs.h"
#include "TPCBase/PadPos.h"

namespace o2 {
namespace TPC {

class PadRegionInfo
{
  public:
    /// default constructor
    PadRegionInfo() {}

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
                  const unsigned char globalRowOffset
                 );

    const unsigned char getRegion() const { return mRegion; }

    const unsigned char getPartition() const { return mPartition; }

    const unsigned char getNumberOfPadRows() const { return mNumberOfPadRows; }

    const unsigned short getNumberOfPads() const { return mNumberOfPads; }

    const float getPadHeight() const { return mPadHeight; }

    const float getPadWidth() const { return mPadWidth; }

    const float getRadiusFirstRow() const { return mRadiusFirstRow; }

    const unsigned char getGlobalRowOffset() const { return mGlobalRowOffset; }
//   const unsigned char  getRowOffset()        const { return mRowOffset;        }
//   const float          getXhelper()         const { return mXhelper;         }

    const unsigned char getPadsInRow(const PadPos &padPos) const
    {
      return mPadsPerRow[padPos.getRow() - mGlobalRowOffset];
    }

    const unsigned char getPadsInRow(const int row) const { return mPadsPerRow[row - mGlobalRowOffset]; }

    const unsigned char getPadsInRowRegion(const int row) const { return mPadsPerRow[row]; }

    //TODO check function
    const bool isInRegion(float localX, float border = 0.f) const
    {
      return localX - mRadiusFirstRow - border > 0.f && localX - mRadiusFirstRow <
                                                        (mNumberOfPadRows + 1) * mPadHeight + border;
    }

    const PadPos findPad(const LocalPosition3D &pos) const;

    const PadPos findPad(const LocalPosition2D &pos, const Side side = Side::A) const;

    const PadPos findPad(const float localX, const float localY, const Side side = Side::A) const;

  private:
    float mPadHeight{0.f};              ///< pad height in this region
    float mPadWidth{0.f};               ///< pad width in this region
    float mRadiusFirstRow{0.f};         ///< radial position of first row
    float mXhelper{0.f};                ///< helper value to calculate pads per row
    unsigned short mNumberOfPads{0};    ///< total number of pads in region
    unsigned char mPartition{0};        ///< partition number
    unsigned char mRegion{0};           ///< pad region number
    unsigned char mNumberOfPadRows{0};  ///< number of rows in region

    unsigned char mRowOffset{0};        ///< row offset in region with same height

    unsigned char mGlobalRowOffset{0};  ///< global pad row offset

    void init();

    std::vector<unsigned char> mPadsPerRow; ///< number of pad in each row
};

}
}
#endif
