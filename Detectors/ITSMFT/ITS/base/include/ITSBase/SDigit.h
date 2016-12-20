/// \file SDigit.h
/// \brief Definition of the ITS summable digit class

#ifndef ALICEO2_ITS_SDIGIT_H
#define ALICEO2_ITS_SDIGIT_H

#include <TObject.h>

namespace AliceO2
{
namespace ITS
{
class SDigit : public TObject
{
 public:
  enum { kBuffSize = 10 };
  //
  SDigit();
  SDigit(Int_t track, Int_t hit, UInt_t chip, UInt_t index, Double_t signal, Int_t roCycle = 0);
  SDigit(UInt_t chip, UInt_t index, Double_t noise, Int_t roCycle = 0);
  SDigit(const SDigit& source);
  SDigit& operator=(const SDigit& source);
  virtual ~SDigit() {}
  Double_t getSignal(Int_t i) const { return ((i >= 0 && i < kBuffSize) ? mSignal[i] : 0.0); }
  Double_t getSignal() const { return mTotalSignal; }
  Double_t getSignalAfterElect() const { return mSignalAfterElect; }
  Double_t getSumSignal() const { return mTotalSignal + mNoise; }
  Double_t getNoise() const { return mNoise; }
  Int_t getNsignals() const { return kBuffSize; }
  void addSignal(Int_t track, Int_t hit, Double_t signal);
  void addSignalAfterElect(Double_t signal) { mSignalAfterElect += signal; }
  void addNoise(Double_t noise) { mNoise += noise; }
  void setNoise(Double_t noise) { mNoise = noise; }
  void setROCycle(Int_t cl) { mROCycle = cl; }
  //
  Int_t getTrack(Int_t i) const { return ((i >= 0 && i < kBuffSize) ? mTrack[i] : 0); }
  Int_t getHit(Int_t i) const { return ((i >= 0 && i < kBuffSize) ? mHits[i] : 0); }
  Int_t getChip() const { return mChip; }
  Int_t getNumOfTracks() const { return mNumOfTracks; }
  Int_t getROCycle() const { return mROCycle; }
  //
  void add(const SDigit* pl);
  void addTo(Int_t fileIndex, const SDigit* pl);
  void shiftIndices(Int_t fileIndex);
  void print(Option_t* option = "") const;
  Int_t read(const char* name) { return TObject::Read(name); }
  //
  virtual Bool_t IsSortable() const { return kTRUE; }
  virtual Bool_t IsEqual(const TObject* obj) const { return GetUniqueID() == obj->GetUniqueID(); }
  virtual Int_t Compare(const TObject* obj) const;
  //
  static Int_t getBuffSize() { return kBuffSize; };
  //
 private:
  UShort_t mChip;             ///< Chip number
  UShort_t mNumOfTracks;      ///< Number of tracks contributing
  Int_t mROCycle;             ///< ReadOut cycle
  Int_t mTrack[kBuffSize];    ///< Track Number
  Int_t mHits[kBuffSize];     ///< Hit number
  Float_t mSignal[kBuffSize]; ///< Signals
  Float_t mTotalSignal;       ///< Total signal (no noise)
  Float_t mNoise;             ///< Total noise, coupling, ...
  Float_t mSignalAfterElect;  ///< Signal after electronics
  //
  ClassDef(SDigit, 1) // Item list of signals and track numbers
};
}
}
#endif
