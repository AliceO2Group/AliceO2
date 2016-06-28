#include <cmath>
#include "PadRegionInfo.h"

namespace AliceO2 {
namespace TPC {


PadRegionInfo::PadRegionInfo(const unsigned char region,
                             const unsigned char numberOfPadRows,
                             const float         padHeight,
                             const float         padWidth,
                             const float         radiusFirstRow,
                             const unsigned char rowOffet,
                             const float         xhelper,
                             const unsigned char globalRowOffset
                            )
  : mRegion{region}
  , mNumberOfPadRows{numberOfPadRows}
  , mPadHeight{padHeight}
  , mPadWidth{padWidth}
  , mRadiusFirstRow{radiusFirstRow}
  , mRowOffet{rowOffet}
  , mXhelper{xhelper}
  , mNumberOfPads{0}
  , mGlobalRowOffset{globalRowOffset}
  , mPadsPerRow{numberOfPadRows}
{
  init();
}

void PadRegionInfo::init()
{

  const float ks=mPadHeight/mPadWidth*tan(1.74532925199432948e-01); // tan(10deg)
  // initialize number of pads per row
  for (int irow=0; irow<mNumberOfPadRows; ++irow) {
     mPadsPerRow[irow]=2.*floor(ks*(irow+mRowOffet)+mXhelper);
     mNumberOfPads+=mPadsPerRow[irow];
  }
}

}
}
