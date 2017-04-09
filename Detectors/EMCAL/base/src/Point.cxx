#include "EMCALBase/Point.h"

ClassImp(o2::EMCAL::Point)

using namespace o2::EMCAL;

void Point::PrintStream(std::ostream &stream) const {
  stream  << "EMCAL point: Track " << GetTrackID() << " in detector segment " << GetDetectorID()
          << " at position (" << GetX() << "|" << GetY() << "|" << GetZ() << "), energy loss " << GetEnergyLoss()
          << ", parent " << mParent << " with energy " << mInitialEnergy;
}

Bool_t Point::operator<(const Point &rhs) const {
  if(mParent != rhs.mParent) return mParent < rhs.mParent;
  return GetDetectorID() < rhs.GetDetectorID();
}

Bool_t Point::operator==(const Point &rhs) const {
  return (GetDetectorID() == GetDetectorID()) && (mParent == rhs.mParent);
}

Point &Point::operator+=(const Point &rhs) {
  SetEnergyLoss(GetEnergyLoss() + rhs.GetEnergyLoss());
  return *this;
}

Point Point::operator+(const Point &rhs) const {
  Point result(*this);
  result.SetEnergyLoss(result.GetEnergyLoss() + rhs.GetEnergyLoss());
  return *this;
}

std::ostream &operator<<(std::ostream &stream, const Point &p) {
  p.PrintStream(stream);
  return stream;
}
