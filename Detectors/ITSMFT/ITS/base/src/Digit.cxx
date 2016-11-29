/// \file Digit.cxx
/// \brief Digit structure for ITS digits

#include "ITSBase/Digit.h"

ClassImp(AliceO2::ITS::Digit)

using namespace AliceO2::ITS;

Digit::Digit() :
  FairTimeStamp(),
  fChipIndex(0),
  fRow(0),
  fCol(0),
  fCharge(0.),
  fLabels()
{
}

Digit::Digit(UShort_t chipindex, UShort_t row, UShort_t col, Double_t charge, Double_t time) :
  FairTimeStamp(time),
  fChipIndex(chipindex),
  fRow(row),
  fCol(col),
  fCharge(charge),
  fLabels()
{
}

Digit::~Digit()
{ }

Digit &Digit::operator+=(const Digit &other)
{
  fCharge += other.fCharge;
  return *this;
}

const Digit Digit::operator+(const Digit &other)
{
  Digit result(*this);
  result += other;
  return result;
}


std::ostream &Digit::Print(std::ostream &output) const
{
  output << "ITS Digit of chip index [" << fChipIndex << "] and pixel [" << fRow << ','<< fCol << "] with charge " << fCharge <<
  " at time stamp" << fTimeStamp;
  return output;
}
