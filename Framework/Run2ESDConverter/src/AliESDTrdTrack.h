#ifndef ALIESDTRDTRACK_H
#define ALIESDTRDTRACK_H

//
// ESD format for the TRD tracks calculated in the
// Global Tracking Unit, used for the TRD L1 trigger
// Author: Jochen Klein <jochen.klein@cern.ch>

#include "TRef.h"

#include "AliVTrdTrack.h"
#include "AliESDTrdTracklet.h"
#include "AliESDtrack.h"

class AliESDTrdTrack : public AliVTrdTrack {

 public:

  AliESDTrdTrack();
  virtual ~AliESDTrdTrack() {};
  AliESDTrdTrack(const AliESDTrdTrack& track);
  AliESDTrdTrack& operator=(const AliESDTrdTrack& track);
  virtual void Copy(TObject &obj) const;

  ULong64_t GetTrackWord(Int_t rev) const;
  ULong64_t GetExtendedTrackWord(Int_t rev) const;

  Int_t GetA()         const { return fA; }
  Int_t GetB()         const { return fB; }
  Int_t GetC()         const { return fC; }
  Int_t GetY()         const { return fY; }
  Int_t GetLayerMask() const { return fLayerMask; }
  Int_t GetPID()       const { return fPID; }
  Int_t GetPt()        const;
  Int_t GetStack()     const { return fStack; }
  Int_t GetSector()    const { return fSector; }
  UChar_t GetFlags()   const { return fFlags; }
  UChar_t GetFlagsTiming() const { return fFlagsTiming; }
  Bool_t GetTrackInTime() const { return (fFlagsTiming & 0x1); }
  Int_t GetLabel()     const { return fLabel; }
  Int_t GetTrackletIndex(const Int_t iLayer) const { return fTrackletIndex[iLayer]; }

  Double_t Pt()        const { return GetPt() / 128.; }
  Double_t Phi()       const { return 0.; };
  Double_t Eta()       const { return 0.; };

  Int_t GetNTracklets() const {
    Int_t count = 0;
    for (Int_t iLayer = 0; iLayer < 6; ++iLayer)
      count += (fLayerMask >> iLayer) & 1;
    return count;
  }
  AliESDTrdTracklet* GetTracklet(Int_t idx) const
    { return (GetLayerMask() & (1<<idx)) ? (AliESDTrdTracklet*) ((fTrackletRefs[idx]).GetObject()) : 0x0; }
  AliVTrack* GetTrackMatch() const { return (AliVTrack*) fTrackMatch.GetObject(); }

  void SetA(Int_t a)            { fA = a; }
  void SetB(Int_t b)            { fB = b; }
  void SetC(Int_t c)            { fC = c; }
  void SetY(Int_t y)            { fY = y; }
  void SetLayerMask(Int_t mask) { fLayerMask = mask; }
  void SetPID(Int_t pid)        { fPID = pid; }
  void SetLabel(Int_t label)    { fLabel = label; }
  void SetSector(Int_t sector)  { fSector = sector; }
  void SetStack(Int_t stack)    { fStack = stack; }
  void SetFlags(Int_t flags)    { fFlags = flags; }
  void SetFlagsTiming(Int_t flags) { fFlagsTiming = flags; }
  void SetReserved(Int_t res)   { fReserved = res; }
  void SetTrackletIndex(const Char_t idx, const Int_t layer) { fTrackletIndex[layer] = idx; }

  void AddTrackletReference(AliESDTrdTracklet* trkl, Int_t layer) { fTrackletRefs[layer] = trkl; }
  void SetTrackMatchReference(AliVTrack *trk) { fTrackMatch = trk; }

  Bool_t IsSortable() const  { return kTRUE; }
  Int_t Compare(const TObject* obj) const;

 protected:

  void AppendBits(ULong64_t &word, UInt_t nBits, UInt_t val) const { word = (word << nBits) | (val & ~(~((ULong64_t) 0) << nBits)); }

  Int_t    fSector;			  // sector in which the track was found
  Char_t   fStack;			  // stack in which the track was found
					  // (unique because of stack-wise tracking)
  Int_t    fA;				  // transverse offset from nominal primary vertex
  Int_t    fB;				  // slope in transverse plane
  Short_t  fC;				  // slope in r-z plane
  Short_t  fY;				  // y position of the track
  UChar_t  fPID;			  // electron PID for this track
  Char_t   fLayerMask;			  // mask of contributing tracklets
  Char_t   fTrackletIndex[fgkNlayers];	  //[fgkNlayers] index to tracklets
  UShort_t fFlags;			  // flags (high-pt, electron, positron)
  UChar_t  fFlagsTiming;                  // timing flags (track in-time, ...)
  UChar_t  fReserved;			  // reserved for future use

  TRef fTrackletRefs[fgkNlayers];         // references to contributing tracklets

  TRef fTrackMatch;                       // reference to matched global track
					  // to reject TRD tracks from late conversions

  Int_t fLabel;				  // Track label

  ClassDef(AliESDTrdTrack,7)
};

#endif
