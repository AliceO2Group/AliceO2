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
  void Fill(int iStep, const Ts&... valuesAndWeight);
  void Fill(int iStep, int nParams, double positionAndWeight[]);

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
  StepTHnT(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxes, Int_t* nBins, std::vector<Double_t> binEdges[], const char** axisTitles);
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
void StepTHn::Fill(int iStep, const Ts&... valuesAndWeight)
{
  constexpr int nArgs = sizeof...(Ts);
  double tempArray[] = {static_cast<double>(valuesAndWeight)...};
  Fill(iStep, nArgs, tempArray);
}

#endif
