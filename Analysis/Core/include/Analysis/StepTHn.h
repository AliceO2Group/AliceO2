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

class TArray;
class TArrayF;
class TArrayD;
class TCollection;

class StepTHnBase : public TNamed
{
 public:
  StepTHnBase() : TNamed() {}
  StepTHnBase(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxis, Int_t* nBins, Double_t** binLimits, const char** axisTitles) : TNamed(name, title) {}

  virtual void Fill(const Double_t* var, Int_t istep, Double_t weight = 1.) = 0;

  virtual THnBase* getTHn(Int_t step, Bool_t sparse = kFALSE) = 0;
  virtual Int_t getNSteps() = 0;
  virtual Int_t getNVar() = 0;

  virtual TArray* getValues(Int_t step) = 0;
  virtual TArray* getSumw2(Int_t step) = 0;

  virtual void deleteContainers() = 0;

  virtual Long64_t Merge(TCollection* list) = 0;

  ClassDef(StepTHnBase, 1) // StepTHn base class
};

// TODO equidistant binning for THn

template <class TemplateArray, typename TemplateType>
class StepTHn : public StepTHnBase
{
 public:
  StepTHn();
  StepTHn(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxis, Int_t* nBins, Double_t** binLimits, const char** axisTitles);
  ~StepTHn() override;

  void Fill(const Double_t* var, Int_t istep, Double_t weight = 1.) override;

  THnBase* getTHn(Int_t step, Bool_t sparse = kFALSE) override
  {
    if (!mTarget || !mTarget[step])
      createTarget(step, sparse);
    return mTarget[step];
  }
  Int_t getNSteps() override { return mNSteps; }
  Int_t getNVar() override { return mNVars; }

  TArray* getValues(Int_t step) override { return mValues[step]; }
  TArray* getSumw2(Int_t step) override { return mSumw2[step]; }

  StepTHn(const StepTHn& c);
  StepTHn& operator=(const StepTHn& corr);
  void Copy(TObject& c) const override;

  Long64_t Merge(TCollection* list) override;

 protected:
  void init();
  void createTarget(Int_t step, Bool_t sparse);
  void deleteContainers() override;

  Long64_t getGlobalBinIndex(const Int_t* binIdx);

  Long64_t mNBins;         // number of total bins
  Int_t mNVars;            // number of variables
  Int_t mNSteps;           // number of selection steps
  TemplateArray** mValues; //[mNSteps] data container
  TemplateArray** mSumw2;  //[mNSteps] data container

  THnBase** mTarget; //! target histogram

  TAxis** mAxisCache;  //! cache axis pointers (about 50% of the time in Fill is spent in GetAxis otherwise)
  Int_t* mNbinsCache;  //! cache Nbins per axis
  Double_t* mLastVars; //! caching of last used bins (in many loops some vars are the same for a while)
  Int_t* mLastBins;    //! caching of last used bins (in many loops some vars are the same for a while)

  THnSparse* mPrototype; // not filled used as prototype histogram for axis functionality etc.

  ClassDef(StepTHn, 1) // THn like container
};

typedef StepTHn<TArrayF, Float_t> StepTHnF;
typedef StepTHn<TArrayD, Double_t> StepTHnD;

#endif
