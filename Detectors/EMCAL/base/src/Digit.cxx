#include <iostream>
#include "EMCALBase/Digit.h"

using namespace o2::EMCAL;

ClassImp(Digit)

Digit::Digit(Int_t module, Int_t tower, Double_t amplitude, Double_t time):
FairTimeStamp(time),
mModule(module),
mTower(tower),
mAmplitude(amplitude)
{
}

bool Digit::operator<(const Digit &other) const {
  return GetTimeStamp() < other.GetTimeStamp();
}

void Digit::PrintStream(std::ostream &stream) const {
  stream << "EMCAL Digit: Module " << mModule <<", Tower " << mTower << ", Time " << GetTimeStamp() << " wiht amplitude " << mAmplitude;
}

std::ostream& operator<<(std::ostream &stream, const Digit &digi){
  digi.PrintStream(stream);
  return stream;
}
