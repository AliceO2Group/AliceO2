#ifndef ALIALGAUX_H
#define ALIALGAUX_H

#include <TMath.h>
#include <TString.h>
class AliCDBId;
class TMap;
class TList;

using namespace TMath;

/*--------------------------------------------------------
  Collection of auxillary methods
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


namespace AliAlgAux {
  const double kAlmostZeroD = 1e-15;
  const float  kAlmostZeroF = 1e-11;
  const double kAlmostOneD = 1.-kAlmostZeroD;
  const float  kAlmostOneF = 1.-kAlmostZeroF;
  const double kTinyDist   = 1.e-7; // ignore distances less that this
  //
  enum {kColl,kCosm,kNTrackTypes};
  //
  inline Double_t Sector2Alpha(int sect);
  inline Int_t    Phi2Sector(double alpha);
  inline Double_t SectorDAlpha()                         {return Pi()/9;}
  //
  template<typename F> void   BringTo02Pi(F &phi);
  template<typename F> void   BringToPiPM(F &phi);
  template<typename F> Bool_t OKforPhiMin(F phiMin,F phi);
  template<typename F> Bool_t OKforPhiMax(F phiMax,F phi);
  template<typename F> F      MeanPhiSmall(F phi0, F phi1);
  template<typename F> F      DeltaPhiSmall(F phi0, F phi1);
  template<typename F> Bool_t SmallerAbs(F d, F tolD)    {return Abs(d)<tolD;}
  template<typename F> Bool_t Smaller(F d, F tolD)       {return d<tolD;}
  //
  inline Int_t  NumberOfBitsSet(UInt_t x);
  inline Bool_t IsZeroAbs(double d) {return SmallerAbs(d,kAlmostZeroD);}
  inline Bool_t IsZeroAbs(float  f) {return SmallerAbs(f,kAlmostZeroF);}
  inline Bool_t IsZeroPos(double d) {return Smaller(d,kAlmostZeroD);}
  inline Bool_t IsZeroPos(float  f) {return Smaller(f,kAlmostZeroF);}
  //
  int    FindKeyIndex(int key, const int *arr, int n);
  //
  void   PrintBits(ULong64_t patt, Int_t maxBits);
  //
  // OCDB related stuff
  void      CleanOCDB();
  AliCDBId* FindCDBId(const TList* cdbList,const TString& key);
  void      RectifyOCDBUri(TString& inp);
  Bool_t    PreloadOCDB(int run, const TMap* cdbMap, const TList* cdbList);
}

//_________________________________________________________________________________
template<typename F>
inline void AliAlgAux::BringTo02Pi(F &phi) {
  // bring phi to 0-2pi range
  if (phi<0) phi+=TwoPi(); else if (phi>TwoPi()) phi-=TwoPi();
}

//_________________________________________________________________________________
template<typename F>
inline void AliAlgAux::BringToPiPM(F &phi) {
  // bring phi to -pi:pi range
  if (phi>Pi()) phi-=TwoPi();
}
//_________________________________________________________________________________
template<typename F>
inline Bool_t AliAlgAux::OKforPhiMin(F phiMin,F phi) {
  // check if phi is above the phiMin, phi's must be in 0-2pi range
  F dphi = phi-phiMin;
  return ((dphi>0 && dphi<Pi()) || dphi<-Pi()) ? kTRUE:kFALSE;
}

//_________________________________________________________________________________
template<typename F>
inline Bool_t AliAlgAux::OKforPhiMax(F phiMax,F phi) {
  // check if phi is below the phiMax, phi's must be in 0-2pi range
  F dphi = phi-phiMax;
  return ((dphi<0 && dphi>-Pi()) || dphi>Pi()) ? kTRUE:kFALSE;
}

//_________________________________________________________________________________
template<typename F>
inline F AliAlgAux::MeanPhiSmall(F phi0, F phi1) {
  // return mean phi, assume phis in 0:2pi
  F phi;
  if (!OKforPhiMin(phi0,phi1)) {phi=phi0; phi0=phi1; phi1=phi;}
  if (phi0>phi1) phi = (phi1 - (TwoPi()-phi0))/2; // wrap
  else           phi = (phi0+phi1)/2;
  BringTo02Pi(phi);
  return phi;
}

//_________________________________________________________________________________
template<typename F>
inline F AliAlgAux::DeltaPhiSmall(F phi0, F phi1) {
  // return delta phi, assume phis in 0:2pi
  F del;
  if (!OKforPhiMin(phi0,phi1)) {del=phi0; phi0=phi1; phi1=del;}
  del = phi1 - phi0;
  if (del<0) del += TwoPi();
  return del;
}

//_________________________________________________________________________________
inline Int_t AliAlgAux::NumberOfBitsSet(UInt_t x) {
  // count number of non-0 bits in 32bit word
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

//_________________________________________________________________________________
inline Double_t AliAlgAux::Sector2Alpha(int sect) {
  // get barrel sector alpha in -pi:pi format
  if (sect>8)sect-=18; return (sect+0.5)*SectorDAlpha();
} 

//_________________________________________________________________________________
inline Int_t AliAlgAux::Phi2Sector(double phi) {
  // get barrel sector from phi in -pi:pi format
  int sect = Nint( (phi*RadToDeg()-10)/20. );
  if (sect<0) sect+=18; 
  return sect;
} 

#endif
