/// \file SDigit.cxx
/// \brief Implementation of the ITSMFT summable digit class

#include <FairLogger.h>
#include <TMath.h>

#include "ITSMFTBase/SDigit.h"

using std::istream;
using std::swap;

ClassImp(o2::ITSMFT::SDigit)

  using namespace o2::ITSMFT;

//______________________________________________________________________
SDigit::SDigit() : mChip(0), mNumOfTracks(0), mROCycle(0), mTotalSignal(0.0), mNoise(0.0), mSignalAfterElect(0.0)
{
  // Default constructor
  for (int i = kBuffSize; i--;) {
    mTrack[i] = -2;
    mHits[i] = -1;
    mSignal[i] = 0;
  }
}

//______________________________________________________________________
SDigit::SDigit(UInt_t chip, UInt_t index, Double_t noise, Int_t roCycle)
  : mChip(chip), mNumOfTracks(0), mROCycle(roCycle), mTotalSignal(0.0), mNoise(noise), mSignalAfterElect(0.0)
{
  // Standard noise constructor
  SetUniqueID(index);
  for (int i = kBuffSize; i--;) {
    mTrack[i] = -2;
    mHits[i] = -1;
    mSignal[i] = 0;
  }
}

//______________________________________________________________________
SDigit::SDigit(Int_t track, Int_t hit, UInt_t chip, UInt_t index, Double_t signal, Int_t roCycle)
  : mChip(chip), mNumOfTracks(1), mROCycle(roCycle), mTotalSignal(signal), mNoise(0.0), mSignalAfterElect(0.0)
{
  // Standard signal constructor
  // Inputs:
  //    Int_t track     The track number which produced this signal
  //    Int_t hit       The hit number which produced this signal
  //    Int_t chip    The chip where this signal occurred
  //    Int_t index     The cell index where this signal occurred
  //    Double_t signal The value of the signal (ionization)
  //    Int_t roCycle   Read-Out cycle
  SetUniqueID(index);
  mTrack[0] = track;
  mHits[0] = hit;
  mSignal[0] = signal;
  for (int i = 1; i < kBuffSize; i++) {
    mTrack[i] = -2;
    mHits[i] = -1;
    mSignal[i] = 0;
  }
}

//______________________________________________________________________
SDigit& SDigit::operator=(const SDigit& source)
{
  // = operator
  if (&source != this) {
    this->~SDigit();
    new (this) SDigit(source);
  }
  return *this;
  //
}

//______________________________________________________________________
SDigit::SDigit(const SDigit& source)
  : TObject(source),
    mChip(source.mChip),
    mNumOfTracks(source.mNumOfTracks),
    mROCycle(source.mROCycle),
    mTotalSignal(source.mTotalSignal),
    mNoise(source.mNoise),
    mSignalAfterElect(source.mSignalAfterElect)
{
  // Copy operator
  for (Int_t i = kBuffSize; i--;) {
    mTrack[i] = source.mTrack[i];
    mSignal[i] = source.mSignal[i];
    mHits[i] = source.mHits[i];
  } // end if i
  //
}

//______________________________________________________________________
void SDigit::addSignal(Int_t track, Int_t hit, Double_t signal)
{
  // Adds this track number and signal to the pList and orders them
  // Inputs:
  //    Int_t track     The track number which produced this signal
  //    Int_t hit       The hit number which produced this signal
  //    Int_t chip    The chip where this signal occurred
  //    Int_t index     The cell index where this signal occurred
  //    Double_t signal The value of the signal (ionization)
  Int_t i, j;
  Bool_t flg = kFALSE;
  //
  if (TMath::Abs(signal) > 2147483647.0) {
    // PH 2147483647 is the max. integer
    // PH This apparently is a problem which needs investigation
    LOG(WARNING) << "Too big or too small signal value " << signal << FairLogger::endl;
    signal = TMath::Sign((Double_t)2147483647, signal);
  }
  //
  mTotalSignal += signal; // Keep track of sum signal.
  for (i = mNumOfTracks; i--;) {
    if (track == mTrack[i]) {
      mSignal[i] += signal;
      flg = kTRUE;
      break;
    } // end for i & if.
  }
  //
  if (flg) {
    if (mNumOfTracks > 1) { // resort arrays.
      for (i = 1; i < mNumOfTracks; i++) {
        j = i;
        while (j > 0 && mSignal[j] > mSignal[j - 1]) {
          std::swap(mTrack[j - 1], mTrack[j]);
          std::swap(mHits[j - 1], mHits[j]);
          std::swap(mSignal[j - 1], mSignal[j]);
          j--;
        } // end while
      }   // end if i
    }     // end if added to existing and resorted array
    return;
  }
  //
  // new entry add it in order.
  if (mNumOfTracks == (kBuffSize - 1) && signal <= mSignal[kBuffSize - 1])
    return; // if this signal is <= smallest then don't add it.
  //
  for (i = mNumOfTracks; i--;) {
    if (signal > mSignal[i]) { // shift smaller signals to the end of the list
      if (i < kBuffSize - 2) { // (if there is a space...)
        mSignal[i + 1] = mSignal[i];
        mTrack[i + 1] = mTrack[i];
        mHits[i + 1] = mHits[i];
      }
    } else {
      mSignal[i + 1] = signal;
      mTrack[i + 1] = track;
      mHits[i + 1] = hit;
      if (mNumOfTracks < kBuffSize - 1)
        mNumOfTracks++;
      return; // put it in the right place, now exit.
    }         //  end if
  }           // end if; end for i
  //
  // Still haven't found the right place. Must be at top of list.
  mSignal[0] = signal;
  mTrack[0] = track;
  mHits[0] = hit;
  mNumOfTracks++;
  return;
}

//______________________________________________________________________
void SDigit::add(const SDigit* pl)
{
  // Adds the contents of pl to this
  // pl could come from different chip and index
  Double_t sigT = 0.0;
  for (int i = pl->getNumOfTracks(); i--;) {
    double sig = pl->getSignal(i);
    addSignal(pl->getTrack(i), pl->getHit(i), sig);
    sigT += sig;
  } // end for i
  mTotalSignal += (pl->mTotalSignal - sigT);
  mNoise += pl->mNoise;
  return;
  //
}

//______________________________________________________________________
void SDigit::addTo(Int_t fileIndex, const SDigit* pl)
{
  // Adds the contents of pl to this with track number off set given by
  // fileIndex.
  // Inputs:
  //    Int_t fileIndex      track number offset value
  //    SDigit *pl  an SDigit to be added to this class.
  //
  for (int i = pl->getNumOfTracks(); i--;)
    addSignal(pl->getTrack(i) + fileIndex, pl->getHit(i), pl->getSignal(i));
  mSignalAfterElect += (pl->mSignalAfterElect + pl->mNoise - mNoise);
  mNoise = pl->mNoise;
}

//______________________________________________________________________
void SDigit::shiftIndices(Int_t fileIndex)
{
  // Shift track numbers
  //
  for (int i = getNumOfTracks(); i--;)
    mTrack[i] += fileIndex;
}

//______________________________________________________________________
Int_t SDigit::Compare(const TObject* obj) const
{
  // compare objects
  if (GetUniqueID() < obj->GetUniqueID())
    return -1;
  if (GetUniqueID() > obj->GetUniqueID())
    return 1;
  return 0;
}

//______________________________________________________________________
void SDigit::print(Option_t*) const
{
  // print itself
  printf("Mod: %4d Index:%7d Ntr:%2d | TotSignal:%.2e Noise:%.2e ROCycle: %d|", mChip, GetUniqueID(), mNumOfTracks,
         mTotalSignal, mNoise, mROCycle);
  for (int i = 0; i < mNumOfTracks; i++)
    printf("%d(%.2e) |", mTrack[i], mSignal[i]);
  printf("\n");
}
