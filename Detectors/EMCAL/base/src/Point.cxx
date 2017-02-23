#include "EMCALBase/Point.h"

ClassImp(AliceO2::EMCAL::Point)

using namespace AliceO2::EMCAL;

Point::Point():
  FairMCPoint()
{
}

Point::Point(Int_t shunt, Int_t primary, Int_t trackID, Int_t parentID, Int_t detID, Int_t initialEnergy, TVector3 pos, TVector3 mom,
             Double_t tof, Double_t eLoss, UInt_t EventId):
  FairMCPoint(trackID, detID, pos, mom, tof, 0., eLoss, EventId),
  mShunt(shunt),
  mPrimary(primary),
  mParent(parentID),
  mInitialEnergy(initialEnergy)
{
  
}

Point::~Point() {

}

void Point::PrintStream(std::ostream &stream) const {
  stream  << "EMCAL point: Track " << fTrackID << " in detector segment " << fDetectorID
          << " at position (" << fX << "|" << fY << "|" << fZ << "), energy loss " << fELoss
          << ", parent " << mParent << " with energy " << mInitialEnergy;
}

Bool_t Point::operator<(const Point &rhs) const {
  if(mParent != rhs.mParent) return mParent < rhs.mParent;
  return fDetectorID < rhs.fDetectorID;
}

Bool_t Point::operator==(const Point &rhs) const {
  return (fDetectorID == rhs.fDetectorID) && (mParent == rhs.mParent);
}

Point &Point::operator+=(const Point &rhs) {
  fELoss += rhs.fELoss;
  return *this;
}

Point Point::operator+(const Point &rhs) const {
  Point result(*this);
  result.fELoss += rhs.fELoss;
  return *this;
}

std::ostream &operator<<(std::ostream &stream, const Point &p) {
  p.PrintStream(stream);
  return stream;
}
