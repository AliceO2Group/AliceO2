/// \file HwFixedPoint.cxx
/// \brief Class to Fixed Point calculations as it would be done in Hardware

#include "TPCSimulation/HwFixedPoint.h"
#include "FairLogger.h"
#include "TMath.h"

ClassImp(AliceO2::TPC::HwFixedPoint)

using namespace AliceO2::TPC;

//________________________________________________________________________
HwFixedPoint::HwFixedPoint(UShort_t totalWidth, UShort_t decPrec):
  TObject(),
  mDecPrecision(decPrec),
  mTotalWidth(totalWidth),
  mValue(0)
{
  mMask = ((T) -1) - ((T)TMath::Power(2,(int)(mTotalWidth))-1);
}

HwFixedPoint::HwFixedPoint(Int_t val, UShort_t totalWidth, UShort_t decPrec):
  TObject(),
  mDecPrecision(decPrec),
  mTotalWidth(totalWidth)
{
  mMask = ((T) -1) - ((T)TMath::Power(2,(int)(mTotalWidth))-1);
  setValue(val * (1 << mDecPrecision));
}

HwFixedPoint::HwFixedPoint(Float_t val, UShort_t totalWidth, UShort_t decPrec):
  TObject(),
  mDecPrecision(decPrec),
  mTotalWidth(totalWidth)
{
  mMask = ((T) -1) - ((T)TMath::Power(2,(int)(mTotalWidth))-1);
  setValue(val * (1 << mDecPrecision));
}

HwFixedPoint::HwFixedPoint(Double_t val, UShort_t totalWidth, UShort_t decPrec):
  TObject(),
  mDecPrecision(decPrec),
  mTotalWidth(totalWidth)
{
  mMask = ((T) -1) - ((T)TMath::Power(2,(int)(mTotalWidth))-1);
  setValue(val * (1 << mDecPrecision));
}

HwFixedPoint::HwFixedPoint(const HwFixedPoint & val):
  TObject(),
  mDecPrecision(val.mDecPrecision),
  mTotalWidth(val.mTotalWidth),
  mMask(val.mMask),
  mValue(val.mValue)
{
}

HwFixedPoint::HwFixedPoint(const HwFixedPoint & val, UShort_t totalWidth, UShort_t decPrec):
  TObject(),
  mDecPrecision(decPrec),
  mTotalWidth(totalWidth)
{
  mMask = ((T) -1) - ((T)TMath::Power(2,(int)(mTotalWidth))-1);
  setValue( ((Double_t) val) * (1 << mDecPrecision));
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
{
}

//________________________________________________________________________
void HwFixedPoint::setValue(T val) {

  if ((val & (1 << (mTotalWidth-1))) >> (mTotalWidth-1) == 1) {
    mValue = mMask | val;
  } else {
    mValue = (~mMask) & val;
  }
}
