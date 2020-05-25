
#include "Analysis/ReducedEvent.h"
#include "Analysis/ReducedTrack.h"

ClassImp(ReducedEvent)

TClonesArray* ReducedEvent::fgTracks = 0;

//____________________________________________________________________________
ReducedEvent::ReducedEvent() :
  fEventTag(0),
  fVtx(),
  fCentVZERO(-999.0),
  fTracks(0x0)
{
  //
  // Constructor
  //
  for(Int_t i=0; i<3; ++i) {fVtx[i]=-999.;}
  if(!fgTracks) fgTracks = new TClonesArray("ReducedTrack", 100000);
    fTracks = fgTracks;
}


//____________________________________________________________________________
ReducedEvent::~ReducedEvent()
{
  //
  // De-Constructor
  //
}

//_____________________________________________________________________________
void ReducedEvent::ClearEvent() {
  //
  // clear the event
  //
  if(fTracks) fTracks->Clear("C");
  fEventTag = 0;
  fCentVZERO = -999.0;
  for(int i=0; i<3; ++i) 
    fVtx[i]=-999.;
}
