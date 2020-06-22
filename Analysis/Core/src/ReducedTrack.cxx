
#include "Analysis/ReducedTrack.h"

ClassImp(ReducedTrack)

//_______________________________________________________________________________
ReducedTrack::ReducedTrack() :
  fTrackId(0),
  fP(),
  fIsCartesian(kFALSE),
  fCharge(0),
  fFlags(0)
{
  //
  // Constructor
  //
  fP[0]=0.0; fP[1]=0.0; fP[2]=0.0;
}

//_______________________________________________________________________________
ReducedTrack::ReducedTrack(const ReducedTrack &c) :
  TObject(c),
  fTrackId(c.fTrackId),
  fIsCartesian(c.IsCartesian()),
  fCharge(c.Charge()),
  fFlags(c.Flags())
{
  //
  // Copy constructor
  //
  if(c.IsCartesian()) {
    fP[0]=c.Px();
    fP[1]=c.Py();
    fP[2]=c.Pz();
  } else {
    fP[0]=c.Pt();
    fP[1]=c.Phi();
    fP[2]=c.Eta();}
}

//_______________________________________________________________________________
ReducedTrack::~ReducedTrack()
{
  //
  // De-Constructor
  //
}
