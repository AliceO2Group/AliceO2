/// \file AliITSUpgradeDigi.cxx
/// \brief Digits structure for ITS digits

#include "include/Digit.h"

ClassImp(AliceO2::ITS::Digit)

using namespace AliceO2::ITS;


Digit::Digit():
FairTimeStamp(),
fChipIndex(-1),
fPixelIndex(-1),
fCharge(0.),
fLabels()
{
}

Digit::Digit(Int_t chipindex, Double_t pixelindex, Double_t charge, Double_t time):
FairTimeStamp(time),
fChipIndex(chipindex),
fPixelIndex(pixelindex),
fCharge(charge),
fLabels()
{
}

Digit::~Digit(){}

Digit &Digit::operator+=(const Digit &other){
  fCharge += other.fCharge;
  return *this;
}

const Digit Digit::operator+(const Digit &other){
  Digit result(*this);
  result += other;
  return result;
}


std::ostream &Digit::Print(std::ostream &output) const{
  output << "ITS Digit of chip index [" << fChipIndex << "] and pixel [" << fPixelIndex << "]with charge " << fCharge << " at time stamp" << fTimeStamp;
  return output;
}
