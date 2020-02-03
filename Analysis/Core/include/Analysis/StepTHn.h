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

  ClassDef(StepTHnBase, 1) // AliTHn base class
};

// TODO equidistant binning for THn

template <class TemplateArray, typename TemplateType>
class StepTHn : public StepTHnBase
{
 public:
  StepTHn();
  StepTHn(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxis, Int_t* nBins, Double_t** binLimits, const char** axisTitles);
  virtual ~StepTHn();

  virtual void Fill(const Double_t* var, Int_t istep, Double_t weight = 1.);

  virtual THnBase* getTHn(Int_t step, Bool_t sparse = kFALSE)
  {
    if (!mTarget || !mTarget[step])
      createTarget(step, sparse);
    return mTarget[step];
  }
  virtual Int_t getNSteps() { return mNSteps; }
  virtual Int_t getNVar() { return mNVars; }

  virtual TArray* getValues(Int_t step) { return mValues[step]; }
  virtual TArray* getSumw2(Int_t step) { return mSumw2[step]; }

  StepTHn(const StepTHn& c);
  StepTHn& operator=(const StepTHn& corr);
  virtual void Copy(TObject& c) const;

  virtual Long64_t Merge(TCollection* list);

 protected:
  void init();
  void createTarget(Int_t step, Bool_t sparse);
  virtual void deleteContainers();

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
