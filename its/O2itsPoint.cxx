#include "O2itsPoint.h"

#include <iostream>
using std::cout;
using std::endl;


// -----   Default constructor   -------------------------------------------
O2itsPoint::O2itsPoint()
  : FairMCPoint(),
    fX_out(0.),
    fY_out(0.),
    fZ_out(0.),
    fPx_out(0.),
    fPy_out(0.),
    fPz_out(0.)
{
}
// -------------------------------------------------------------------------

// -----   Standard constructor   ------------------------------------------
O2itsPoint::O2itsPoint(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom,
                                   Double_t startTime, Double_t time, Double_t length, 
                                   Double_t eLoss, Int_t shunt)
  : FairMCPoint(trackID, detID, pos, mom, time, length, eLoss)
{
}
// -------------------------------------------------------------------------

// -----   Destructor   ----------------------------------------------------
O2itsPoint::~O2itsPoint() { }
// -------------------------------------------------------------------------

// -----   Public method Print   -------------------------------------------
void O2itsPoint::Print(const Option_t* opt) const
{
  cout << "-I- O2itsPoint: O2its point for track " << fTrackID
       << " in detector " << fDetectorID << endl;
  cout << "    Position (" << fX << ", " << fY << ", " << fZ
       << ") cm" << endl;
  cout << "    Momentum (" << fPx << ", " << fPy << ", " << fPz
       << ") GeV" << endl;
  cout << "    Time " << fTime << " ns,  Length " << fLength
       << " cm,  Energy loss " << fELoss*1.0e06 << " keV" << endl;
}
// -------------------------------------------------------------------------

ClassImp(O2itsPoint)

