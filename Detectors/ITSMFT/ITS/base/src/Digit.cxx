/// \file Digit.cxx
/// \brief Implementation of the ITS digit

#include "ITSBase/Digit.h"

ClassImp(AliceO2::ITS::Digit)

  using namespace AliceO2::ITS;

Digit::Digit() : FairTimeStamp(), mChipIndex(0), mRow(0), mCol(0), mCharge(0.), mLabels{ -1, -1, -1 } {}
Digit::Digit(UShort_t chipindex, UShort_t row, UShort_t col, Double_t charge, Double_t time)
  : FairTimeStamp(time), mChipIndex(chipindex), mRow(row), mCol(col), mCharge(charge), mLabels{ -1, -1, -1 }
{
}

Digit::~Digit() {}

Digit& Digit::operator+=(const Digit& other)
{
  mCharge += other.mCharge;
  return *this;
}

const Digit Digit::operator+(const Digit& other)
{
  Digit result(*this);
  result += other;
  return result;
}

std::ostream& Digit::print(std::ostream& output) const
{
  output << "ITS Digit of chip index [" << mChipIndex << "] and pixel [" << mRow << ',' << mCol << "] with charge "
         << mCharge << " at time stamp" << fTimeStamp;
  return output;
}
