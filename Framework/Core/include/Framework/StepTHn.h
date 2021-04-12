// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef StepTHn_H
#define StepTHn_H

// optimized data container reusing functionality of THn
// A THnSparse is used to have the axis functionality "for free"

#include "TNamed.h"
#include "THnSparse.h"
#include "TAxis.h"
#include "TArray.h"

#include "Framework/Logger.h"

class TArray;
class TArrayF;
class TArrayD;
class TCollection;

class StepTHn : public TNamed
{
 public:
  StepTHn();
  StepTHn(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxes);
  ~StepTHn() override;

  template <typename... Ts>
  void Fill(Int_t istep, const Ts&... valuesAndWeight);

  THnBase* getTHn(Int_t step, Bool_t sparse = kFALSE)
  {
    if (!mTarget || !mTarget[step]) {
      createTarget(step, sparse);
    }
    return mTarget[step];
  }
  Int_t getNSteps() { return mNSteps; }
  Int_t getNVar() { return mNVars; }

  TArray* getValues(Int_t step) { return mValues[step]; }
  TArray* getSumw2(Int_t step) { return mSumw2[step]; }

  StepTHn(const StepTHn& c);
  StepTHn& operator=(const StepTHn& corr);
  void Copy(TObject& c) const override;

  virtual Long64_t Merge(TCollection* list) = 0;

  TAxis* GetAxis(int i) { return mPrototype->GetAxis(i); }
  void Sumw2(){}; // TODO: added for compatibiltiy with registry, but maybe it would be useful also in StepTHn as toggle for error weights

 protected:
  void init();
  virtual TArray* createArray(const TArray* src = nullptr) const = 0;
  void createTarget(Int_t step, Bool_t sparse);
  void deleteContainers();

  Long64_t getGlobalBinIndex(const Int_t* binIdx);

  Long64_t mNBins;  // number of total bins
  Int_t mNVars;     // number of variables
  Int_t mNSteps;    // number of selection steps
  TArray** mValues; //[mNSteps] data container
  TArray** mSumw2;  //[mNSteps] data container

  THnBase** mTarget; //! target histogram

  TAxis** mAxisCache;  //! cache axis pointers (about 50% of the time in Fill is spent in GetAxis otherwise)
  Int_t* mNbinsCache;  //! cache Nbins per axis
  Double_t* mLastVars; //! caching of last used bins (in many loops some vars are the same for a while)
  Int_t* mLastBins;    //! caching of last used bins (in many loops some vars are the same for a while)

  THnSparse* mPrototype; // not filled used as prototype histogram for axis functionality etc.

  ClassDef(StepTHn, 1) // THn like container
};

template <class TemplateArray>
class StepTHnT : public StepTHn
{
 public:
  StepTHnT() : StepTHn() {}
  StepTHnT(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxes, Int_t* nBins, std::vector<Double_t> binLimits[], const char** axisTitles);
  StepTHnT(const char* name, const char* title, const int nSteps, const int nAxes, const int* nBins, const double* xmin, const double* xmax);
  ~StepTHnT() override = default;

 protected:
  TArray* createArray(const TArray* src = nullptr) const override
  {
    if (src == nullptr) {
      return new TemplateArray(mNBins);
    } else {
      return new TemplateArray(*((TemplateArray*)src));
    }
  }

  Long64_t Merge(TCollection* list) override;

  ClassDef(StepTHnT, 1) // THn like container
};

typedef StepTHnT<TArrayF> StepTHnF;
typedef StepTHnT<TArrayD> StepTHnD;

template <typename... Ts>
void StepTHn::Fill(Int_t istep, const Ts&... valuesAndWeight)
{
  if (istep >= mNSteps) {
    LOGF(FATAL, "Selected step for filling is not in range of StepTHn.");
  }

  constexpr int nParams = sizeof...(Ts);
  // TODO Find a way to avoid the array
  double tempArray[nParams] = {static_cast<double>(valuesAndWeight)...};

  double weight = 1.0;
  if (nParams == mNVars + 1) {
    weight = tempArray[mNVars];
  } else if (nParams != mNVars) {
    LOGF(FATAL, "Fill called with invalid number of parameters (%d vs %d)", mNVars, nParams);
  }

  // fill axis cache
  if (!mAxisCache) {
    mAxisCache = new TAxis*[mNVars];
    mNbinsCache = new Int_t[mNVars];
    for (Int_t i = 0; i < mNVars; i++) {
      mAxisCache[i] = mPrototype->GetAxis(i);
      mNbinsCache[i] = mAxisCache[i]->GetNbins();
    }

    mLastVars = new Double_t[mNVars];
    mLastBins = new Int_t[mNVars];

    // initial values to prevent checking for 0 below
    for (Int_t i = 0; i < mNVars; i++) {
      mLastVars[i] = tempArray[i];
      mLastBins[i] = mAxisCache[i]->FindBin(mLastVars[i]);
    }
  }

  // calculate global bin index
  Long64_t bin = 0;
  for (Int_t i = 0; i < mNVars; i++) {
    bin *= mNbinsCache[i];

    Int_t tmpBin = 0;
    if (mLastVars[i] == tempArray[i]) {
      tmpBin = mLastBins[i];
    } else {
      tmpBin = mAxisCache[i]->FindBin(tempArray[i]);
      mLastBins[i] = tmpBin;
      mLastVars[i] = tempArray[i];
    }
    //Printf("%d", tmpBin);

    // under/overflow not supported
    if (tmpBin < 1 || tmpBin > mNbinsCache[i]) {
      return;
    }

    // bins start from 0 here
    bin += tmpBin - 1;
    //     Printf("%lld", bin);
  }

  if (!mValues[istep]) {
    mValues[istep] = createArray();
    LOGF(info, "Created values container for step %d", istep);
  }

  if (weight != 1.) {
    // initialize with already filled entries (which have been filled with weight == 1), in this case mSumw2 := mValues
    if (!mSumw2[istep]) {
      mSumw2[istep] = createArray();
      LOGF(info, "Created sumw2 container for step %d", istep);
    }
  }

  // TODO probably slow; add StepTHnT::add ?
  mValues[istep]->SetAt(mValues[istep]->GetAt(bin) + weight, bin);
  if (mSumw2[istep]) {
    mSumw2[istep]->SetAt(mSumw2[istep]->GetAt(bin) + weight, bin);
  }
}

#endif
