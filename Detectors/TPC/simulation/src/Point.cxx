#include "TPCSimulation/Point.h"

using std::cout;
using std::endl;

using namespace AliceO2::TPC;

// -----   Default constructor   -------------------------------------------
Point::Point()
  : FairMCPoint()
{
}
// -------------------------------------------------------------------------

// -----   Standard constructor   ------------------------------------------
Point::Point(Int_t trackID, Int_t detID,
                                   TVector3 pos, TVector3 mom,
                                   Double_t tof, Double_t length,
                                   Double_t eLoss)
  : FairMCPoint(trackID, detID, pos, mom, tof, length, eLoss)
{
}
// -------------------------------------------------------------------------

// -----   Destructor   ----------------------------------------------------
Point::~Point() = default;
// -------------------------------------------------------------------------

// -----   Public method Print   -------------------------------------------
void Point::Print(const Option_t* opt) const
{
  cout << "-I- Point: O2tpc point for track " << fTrackID
       << " in detector " << fDetectorID << endl;
  cout << "    Position (" << fX << ", " << fY << ", " << fZ
       << ") cm" << endl;
  cout << "    Momentum (" << fPx << ", " << fPy << ", " << fPz
       << ") GeV" << endl;
  cout << "    Time " << fTime << " ns,  Length " << fLength
       << " cm,  Energy loss " << fELoss*1.0e06 << " keV" << endl;
}
// -------------------------------------------------------------------------

ClassImp(Point)
