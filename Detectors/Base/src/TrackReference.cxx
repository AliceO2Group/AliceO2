/// \file TrackReference.cxx
/// \brief Implementation of the TrackReference class
/// \author Sylwester Radomski (S.Radomski@gsi.de) GSI, Jan 31, 2003

#include "DetectorsBase/TrackReference.h"
#include "TVirtualMC.h"  // for TVirtualMC, gMC
#include <Riostream.h>

using std::endl;
using std::cout;

using namespace o2::Base;

ClassImp(o2::Base::TrackReference)

TrackReference::TrackReference()
  : TObject(),
    mTrackNumber(0),
    mReferencePositionX(0),
    mReferencePositionY(0),
    mReferencePositionZ(0),
    mMomentumX(0),
    mMomentumY(0),
    mMomentumZ(0),
    mTrackLength(0),
    mTof(0),
    mUserId(0),
    mDetectorId(-999)
{
  //
  // Default constructor
  // Creates empty object

  for (Int_t i = 0; i < 16; i++) {
    ResetBit(BIT(i));
  }
}

TrackReference::TrackReference(const TrackReference &tr)
  : TObject(tr),
    mTrackNumber(tr.mTrackNumber),
    mReferencePositionX(tr.mReferencePositionX),
    mReferencePositionY(tr.mReferencePositionY),
    mReferencePositionZ(tr.mReferencePositionZ),
    mMomentumX(tr.mMomentumX),
    mMomentumY(tr.mMomentumY),
    mMomentumZ(tr.mMomentumZ),
    mTrackLength(tr.mTrackLength),
    mTof(tr.mTof),
    mUserId(tr.mUserId),
    mDetectorId(tr.mDetectorId)
{
  // Copy Constructor
}

TrackReference::TrackReference(Int_t label, Int_t id)
  : TObject(),
    mTrackNumber(label),
    mReferencePositionX(0),
    mReferencePositionY(0),
    mReferencePositionZ(0),
    mMomentumX(0),
    mMomentumY(0),
    mMomentumZ(0),
    mTrackLength(TVirtualMC::GetMC()->TrackLength()),
    mTof(TVirtualMC::GetMC()->TrackTime()),
    mUserId(0),
    mDetectorId(id)
{
  //
  // Create Reference object out of label and
  // data in TVirtualMC object
  //
  // Creates an object and fill all parameters
  // from data in VirtualMC
  //

  //

  Double_t vec[4];

  TVirtualMC::GetMC()->TrackPosition(vec[0], vec[1], vec[2]);

  mReferencePositionX = vec[0];
  mReferencePositionY = vec[1];
  mReferencePositionZ = vec[2];

  TVirtualMC::GetMC()->TrackMomentum(vec[0], vec[1], vec[2], vec[3]);

  mMomentumX = vec[0];
  mMomentumY = vec[1];
  mMomentumZ = vec[2];

  // Set Up status code
  // Copy Bits from virtual MC

  for (Int_t i = 14; i < 22; i++) {
    ResetBit(BIT(i));
  }

  SetBit(BIT(14), TVirtualMC::GetMC()->IsNewTrack());
  SetBit(BIT(15), TVirtualMC::GetMC()->IsTrackAlive());
  SetBit(BIT(16), TVirtualMC::GetMC()->IsTrackDisappeared());
  SetBit(BIT(17), TVirtualMC::GetMC()->IsTrackEntering());
  SetBit(BIT(18), TVirtualMC::GetMC()->IsTrackExiting());
  SetBit(BIT(19), TVirtualMC::GetMC()->IsTrackInside());
  SetBit(BIT(20), TVirtualMC::GetMC()->IsTrackOut());
  SetBit(BIT(21), TVirtualMC::GetMC()->IsTrackStop());

  // This particle has to be kept
}

// AliExternalTrackParam * TrackReference::MakeTrack(const TrackReference *ref, Double_t mass)
// {
//   //
//   // Make dummy track from the track reference
//   // negative mass means opposite charge
//   //
//   Double_t xx[5];
//   Double_t cc[15];
//   for (Int_t i=0;i<15;i++) cc[i]=0;
//   Double_t x = ref->X(), y = ref->Y(), z = ref->Z();
//   Double_t alpha = TMath::ATan2(y,x);
//   Double_t xr = TMath::Sqrt(x*x+y*y);
//   xx[0] = ref->LocalY();
//   xx[1] = z;
//   xx[3] = ref->Pz()/ref->Pt();
//   xx[4] = 1./ref->Pt();
//   if (mass<0) xx[4]*=-1.;  // negative mass - negative direction
//   Double_t alphap = TMath::ATan2(ref->Py(),ref->Px())-alpha;
//   if (alphap> TMath::Pi()) alphap-=TMath::Pi();
//   if (alphap<-TMath::Pi()) alphap+=TMath::Pi();
//   xx[2] = TMath::Sin(alphap);
//
//   AliExternalTrackParam * track = new  AliExternalTrackParam(xr,alpha,xx,cc);
//   return track;
// }

void TrackReference::Print(Option_t * /*opt*/) const
{
  cout << Form("Label %d P=%7.2f (PX,PY,PZ)=(%7.2f,%7.2f,%7.2f) (X,Y,Z)=(%7.2f,%7.2f,%7.2f)"
                 " Length=%7.2f Time=%7.2f UserId=%d",
               Label(), P(), Px(), Py(), Pz(), X(), Y(), Z(), GetLength(), GetTime(), UserId())
  << endl;
}
