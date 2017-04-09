/// \file HwFixedPoint.cxx
/// \brief Class to Fixed Point calculations as it would be done in Hardware

#include "TPCSimulation/HwFixedPoint.h"

using namespace o2::TPC;

//________________________________________________________________________
HwFixedPoint::HwFixedPoint(unsigned short totalWidth, unsigned short decPrec)
  : mDecPrecision(decPrec)
  , mTotalWidth(totalWidth)
  , mValue(0)
{
  mMask = ~((1<<mTotalWidth) -1); // ((T) -1) - ((T)exp2((int)(mTotalWidth))-1);
}

HwFixedPoint::HwFixedPoint(int val, unsigned short totalWidth, unsigned short decPrec)
  : mDecPrecision(decPrec)
  , mTotalWidth(totalWidth)
{
  mMask = ~((1<<mTotalWidth) -1); // ((T) -1) - ((T)exp2((int)(mTotalWidth))-1);
  setValue(val * (1 << mDecPrecision));
}

HwFixedPoint::HwFixedPoint(float val, unsigned short totalWidth, unsigned short decPrec)
  : mDecPrecision(decPrec)
  , mTotalWidth(totalWidth)
{
  mMask = ~((1<<mTotalWidth) -1); // ((T) -1) - ((T)exp2((int)(mTotalWidth))-1);
  setValue(val * (1 << mDecPrecision));
}

HwFixedPoint::HwFixedPoint(double val, unsigned short totalWidth, unsigned short decPrec)
  : mDecPrecision(decPrec)
  , mTotalWidth(totalWidth)
{
  mMask = ~((1<<mTotalWidth) -1); // ((T) -1) - ((T)exp2((int)(mTotalWidth))-1);
  setValue(val * (1 << mDecPrecision));
}

HwFixedPoint::HwFixedPoint(const HwFixedPoint & val)
  : mDecPrecision(val.mDecPrecision)
  , mTotalWidth(val.mTotalWidth)
  , mMask(val.mMask)
  , mValue(val.mValue)
{
}

HwFixedPoint::HwFixedPoint(const HwFixedPoint & val, unsigned short totalWidth, unsigned short decPrec)
  : mDecPrecision(decPrec)
  , mTotalWidth(totalWidth)
{
  mMask = ~((1<<mTotalWidth) -1); // ((T) -1) - ((T)exp2((int)(mTotalWidth))-1);
  setValue( ((double) val) * (1 << mDecPrecision));
}


//HwFixedPoint::HwFixedPoint(const HwFixedPoint& other):
//  TObject(other),
//  mDecPrecision(other.mDecPrecision),
//  mTotalWidth(other.mTotalWidth),
//  mMask(other.mMask),
//  mValue(other.mValue)
//{
//}

//________________________________________________________________________
HwFixedPoint::~HwFixedPoint()
= default;

//________________________________________________________________________
void HwFixedPoint::setValue(T val) {

  if ((val & (1 << (mTotalWidth-1))) >> (mTotalWidth-1) == 1) {
    mValue = mMask | val;
  } else {
    mValue = (~mMask) & val;
  }
}
