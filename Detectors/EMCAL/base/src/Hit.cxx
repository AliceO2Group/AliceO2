#include "EMCALBase/Hit.h"

ClassImp(o2::EMCAL::Hit)

using namespace o2::EMCAL;

void Hit::PrintStream(std::ostream &stream) const {
  stream  << "EMCAL point: Track " << GetTrackID() << " in detector segment " << GetDetectorID()
          << " at position (" << GetX() << "|" << GetY() << "|" << GetZ() << "), energy loss " << GetEnergyLoss()
          << ", parent " << mParent << " with energy " << mInitialEnergy;
}

Bool_t Hit::operator<(const Hit &rhs) const {
  if(mParent != rhs.mParent) return mParent < rhs.mParent;
  return GetDetectorID() < rhs.GetDetectorID();
}

Bool_t Hit::operator==(const Hit &rhs) const {
  return (GetDetectorID() == GetDetectorID()) && (mParent == rhs.mParent);
}

Hit &Hit::operator+=(const Hit &rhs) {
  SetEnergyLoss(GetEnergyLoss() + rhs.GetEnergyLoss());
  return *this;
}

Hit Hit::operator+(const Hit &rhs) const {
  Hit result(*this);
  result.SetEnergyLoss(result.GetEnergyLoss() + rhs.GetEnergyLoss());
  return *this;
}

std::ostream &operator<<(std::ostream &stream, const Hit &p) {
  p.PrintStream(stream);
  return stream;
}
