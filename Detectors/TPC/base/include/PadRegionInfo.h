#ifndef AliceO2_TPC_PadRegionInfo_H
#define AliceO2_TPC_PadRegionInfo_H

#include <vector>

#include "PadPos.h"

namespace AliceO2 {
namespace TPC {

class PadRegionInfo {
public:
  PadRegionInfo() {}
  PadRegionInfo(const unsigned char region,
                const unsigned char numberOfPadRows,
                const float         padHeight,
                const float         padWidth,
                const float         radiusFirstRow,
                const unsigned char rowOffet,
                const float         xhelper,
                const unsigned char globalRowOffset
               );

  const unsigned char  getRegion()          const { return mRegion;          }
  const unsigned char  getNumberOfPadRows() const { return mNumberOfPadRows; }
  const unsigned short getNumberOfPads()    const { return mNumberOfPads;    }
  const float          getPadHeight()       const { return mPadHeight;       }
  const float          getPadWidth()        const { return mPadWidth;        }
  const float          getRadiusFirstRow()  const { return mRadiusFirstRow;  }
  const unsigned char  getRowOffet()        const { return mRowOffet;        }
  const float          getXhelper()         const { return mXhelper;         }

  const unsigned char getPadsInRow      (const PadPos &padPos) const { return mPadsPerRow[padPos.getRow()-mGlobalRowOffset]; }
  const unsigned char getPadsInRow      (const int row)        const { return mPadsPerRow[row-mGlobalRowOffset]; }
  const unsigned char getPadsInRowRegion(const int row)        const { return mPadsPerRow[row]; }


private:
  unsigned char  mRegion{0};           /// pad region number
  unsigned char  mNumberOfPadRows{0};  /// number of rows in region
  float          mPadHeight{0.f};      /// pad height in this region
  float          mPadWidth{0.f};       /// pad width in this region

  float          mRadiusFirstRow{0.f}; /// radial position of first row
  unsigned char  mRowOffet{0};         /// row offset in region with same height
  float          mXhelper{0.f};        /// helper value to calculate pad per row

  unsigned char  mGlobalRowOffset{0};  /// global pad row offset

  unsigned short mNumberOfPads{0};     /// total number of pads in region

  void init();
  std::vector<unsigned char> mPadsPerRow{}; /// number of pad in each row
};

}
}
#endif
