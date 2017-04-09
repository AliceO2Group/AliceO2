#include <cmath>

#include "TPCBase/PadRegionInfo.h"

namespace o2 {
namespace TPC {


PadRegionInfo::PadRegionInfo(const unsigned char region,
                             const unsigned char partition,
                             const unsigned char numberOfPadRows,
                             const float         padHeight,
                             const float         padWidth,
                             const float         radiusFirstRow,
                             const unsigned char rowOffset,
                             const float         xhelper,
                             const unsigned char globalRowOffset
                            )
  : mRegion{region}
  , mPartition{partition}
  , mNumberOfPadRows{numberOfPadRows}
  , mPadHeight{padHeight}
  , mPadWidth{padWidth}
  , mRadiusFirstRow{radiusFirstRow}
  , mRowOffset{rowOffset}
  , mXhelper{xhelper}
  , mNumberOfPads{0}
  , mGlobalRowOffset{globalRowOffset}
  , mPadsPerRow(numberOfPadRows)
{
  init();
}

void PadRegionInfo::init()
{

  const float ks=mPadHeight/mPadWidth*tan(1.74532925199432948e-01); // tan(10deg)
  // initialize number of pads per row
  for (unsigned char irow=0; irow<mNumberOfPadRows; ++irow) {
     mPadsPerRow[irow]=2*std::floor(ks*(irow+mRowOffset)+mXhelper);
     mNumberOfPads+=mPadsPerRow[irow];
  }
}

const PadPos PadRegionInfo::findPad(const LocalPosition3D& pos) const
{
  return findPad(pos.getX(), pos.getY(), (pos.getZ()>=0) ? Side::A : Side::C);
}

const PadPos PadRegionInfo::findPad(const LocalPosition2D& pos, const Side side/*=Side::A*/) const
{
  return findPad(pos.getX(), pos.getY(), side);
}

const PadPos PadRegionInfo::findPad(const float localX, const float localY, const Side side/*=Side::A*/) const
{
  if (!isInRegion(localX)) return PadPos(255, 255);

  // the pad coordinate system is for pad-side view.
  // on the A-Side one looks from the back-side, therefore
  // the localY-sign must be changed
  const float localYfactor=(side==Side::A)?-1.f:1.f;
  const unsigned int row  = std::floor((localX-mRadiusFirstRow)/mPadHeight);
  if (row<0 || row>= mNumberOfPadRows) return PadPos(255, 255);

  const unsigned int npads=getPadsInRowRegion(row);
  const unsigned int pad  =int((npads/2*mPadWidth-localYfactor*localY)/mPadWidth);

  if (pad<0 || pad>=npads) return PadPos(255, 255);

  return PadPos(row, pad);
}

}
}
