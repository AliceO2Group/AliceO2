// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Use StepTHn instead of THn and your memory consumption will be drastically reduced
// Once you have the merged output, use getTHn() to get a standard histogram
//
// this storage container is optimized for small memory usage
//   under/over flow bins do not exist
//   sumw2 structure is float only and only create when the weight != 1
//
// Templated version allows also the use of double as storage container

#include "Framework/StepTHn.h"
#include "TList.h"
#include "TCollection.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "THn.h"
#include "TMath.h"

ClassImp(StepTHn);
templateClassImp(StepTHnT);

StepTHn::StepTHn() : mNBins(0),
                     mNVars(0),
                     mNSteps(0),
                     mValues(nullptr),
                     mSumw2(nullptr),
                     mTarget(nullptr),
                     mAxisCache(nullptr),
                     mNbinsCache(nullptr),
                     mLastVars(nullptr),
                     mLastBins(nullptr),
                     mPrototype(nullptr)
{
  // Default constructor (for streaming)
}

StepTHn::StepTHn(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxes) : TNamed(name, title),
                                                                                                   mNBins(0),
                                                                                                   mNVars(nAxes),
                                                                                                   mNSteps(nSteps),
                                                                                                   mValues(nullptr),
                                                                                                   mSumw2(nullptr),
                                                                                                   mTarget(nullptr),
                                                                                                   mAxisCache(nullptr),
                                                                                                   mNbinsCache(nullptr),
                                                                                                   mLastVars(nullptr),
                                                                                                   mLastBins(nullptr),
                                                                                                   mPrototype(nullptr)
{
  // Constructor
  //
  // This will create a container for <nSteps> steps. The memory for such a step is only allocated once the first value is filled.
  // Therefore you can easily create many steps which are only filled under certain analysis settings.
  // For each step a <nAxes> dimensional histogram is created.
  // The axis have <nBins[i]> bins. The bin edges are given in <binEdges[i]>. If there are only two bin edges, equidistant binning is set.

  init();
}

// root-like constructor
template <class TemplateArray>
StepTHnT<TemplateArray>::StepTHnT(const char* name, const char* title, const int nSteps, const int nAxes, const int* nBins, const double* xmin, const double* xmax) : StepTHn(name, title, nSteps, nAxes)
{
  mNBins = 1;
  for (Int_t i = 0; i < mNVars; i++) {
    mNBins *= nBins[i];
  }
  if (mNBins > 250000000) {
    LOGF(warning, "StepTHn: Requesting more than 250M bins (%lld). This will need extensive memory.", mNBins);
  }
  mPrototype = new THnSparseT<TemplateArray>(Form("%s_sparse", name), title, nAxes, nBins, xmin, xmax);
}

template <class TemplateArray>
StepTHnT<TemplateArray>::StepTHnT(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxes,
                                  Int_t* nBins, std::vector<Double_t> binEdges[], const char** axisTitles) : StepTHn(name, title, nSteps, nAxes)
{
  mNBins = 1;
  for (Int_t i = 0; i < mNVars; i++) {
    mNBins *= nBins[i];
  }
  if (mNBins > 250000000) {
    LOGF(warning, "StepTHn: Requesting more than 250M bins (%lld). This will need extensive memory.", mNBins);
  }
  mPrototype = new THnSparseT<TemplateArray>(Form("%s_sparse", name), title, nAxes, nBins);

  for (Int_t i = 0; i < mNVars; i++) {
    if (nBins[i] + 1 == binEdges[i].size()) { // variable-width binning
      mPrototype->GetAxis(i)->Set(nBins[i], &(binEdges[i])[0]);
    } else if (binEdges[i].size() == 2) { // equidistant binning
      mPrototype->GetAxis(i)->Set(nBins[i], binEdges[i][0], binEdges[i][1]);
    } else {
      LOGF(fatal, "Invalid binning information for axis %d with %d bins and %d entries for bin edges", i, nBins[i], binEdges[i].size());
    }
    LOGF(debug, "Histogram %s Axis %d created with %d bins and %d edges", name, i, nBins[i], binEdges[i].size());
    mPrototype->GetAxis(i)->SetTitle(axisTitles[i]);
  }
}

void StepTHn::init()
{
  // initialize

  mValues = new TArray*[mNSteps];
  mSumw2 = new TArray*[mNSteps];

  for (Int_t i = 0; i < mNSteps; i++) {
    mValues[i] = nullptr;
    mSumw2[i] = nullptr;
  }
}

StepTHn::StepTHn(const StepTHn& c) : mNBins(c.mNBins),
                                     mNVars(c.mNVars),
                                     mNSteps(c.mNSteps),
                                     mValues(new TArray*[c.mNSteps]),
                                     mSumw2(new TArray*[c.mNSteps]),
                                     mTarget(nullptr),
                                     mAxisCache(nullptr),
                                     mNbinsCache(nullptr),
                                     mLastVars(nullptr),
                                     mLastBins(nullptr),
                                     mPrototype(nullptr)
{
  //
  // StepTHn copy constructor
  //

  ((StepTHn&)c).Copy(*this);
}

StepTHn::~StepTHn()
{
  // Destructor

  deleteContainers();

  delete[] mValues;
  delete[] mSumw2;
  delete[] mTarget;
  delete[] mAxisCache;
  delete[] mNbinsCache;
  delete[] mLastVars;
  delete[] mLastBins;
  delete mPrototype;
}

void StepTHn::deleteContainers()
{
  // delete data containers

  for (Int_t i = 0; i < mNSteps; i++) {
    if (mValues && mValues[i]) {
      delete mValues[i];
      mValues[i] = nullptr;
    }

    if (mSumw2 && mSumw2[i]) {
      delete mSumw2[i];
      mSumw2[i] = nullptr;
    }

    if (mTarget && mTarget[i]) {
      delete mTarget[i];
      mTarget[i] = nullptr;
    }
  }
}

StepTHn& StepTHn::operator=(const StepTHn& c)
{
  // assigment operator

  if (this != &c) {
    ((StepTHn&)c).Copy(*this);
  }

  return *this;
}

void StepTHn::Copy(TObject& c) const
{
  // copy function

  StepTHn& target = (StepTHn&)c;

  TNamed::Copy(c);

  target.mNBins = mNBins;
  target.mNVars = mNVars;
  target.mNSteps = mNSteps;

  target.init();

  for (Int_t i = 0; i < mNSteps; i++) {
    if (mValues[i]) {
      target.mValues[i] = createArray(mValues[i]);
    } else {
      target.mValues[i] = nullptr;
    }

    if (mSumw2[i]) {
      target.mSumw2[i] = createArray(mSumw2[i]);
    } else {
      target.mSumw2[i] = nullptr;
    }
  }

  if (mPrototype) {
    target.mPrototype = dynamic_cast<THnSparse*>(mPrototype->Clone());
  }
}

template <class TemplateArray>
Long64_t StepTHnT<TemplateArray>::Merge(TCollection* list)
{
  // Merge a list of StepTHn objects with this (needed for PROOF).
  // Returns the number of merged objects (including this).

  if (!list) {
    return 0;
  }

  if (list->IsEmpty()) {
    return 1;
  }

  TIterator* iter = list->MakeIterator();
  TObject* obj;

  Int_t count = 0;
  while ((obj = iter->Next())) {

    StepTHnT<TemplateArray>* entry = dynamic_cast<StepTHnT<TemplateArray>*>(obj);
    if (entry == nullptr) {
      continue;
    }

    for (Int_t i = 0; i < mNSteps; i++) {
      if (entry->mValues[i]) {
        if (!mValues[i]) {
          mValues[i] = createArray();
        }

        auto source = dynamic_cast<TemplateArray*>(entry->mValues[i])->GetArray();
        auto target = dynamic_cast<TemplateArray*>(mValues[i])->GetArray();
        for (Long64_t l = 0; l < mNBins; l++) {
          target[l] += source[l];
        }
      }

      if (entry->mSumw2[i]) {
        if (!mSumw2[i]) {
          mSumw2[i] = createArray();
        }

        auto source = dynamic_cast<TemplateArray*>(entry->mSumw2[i])->GetArray();
        auto target = dynamic_cast<TemplateArray*>(mSumw2[i])->GetArray();
        for (Long64_t l = 0; l < mNBins; l++) {
          target[l] += source[l];
        }
      }
    }

    count++;
  }

  return count + 1;
}

Long64_t StepTHn::getGlobalBinIndex(const Int_t* binIdx)
{
  // calculates global bin index
  // binIdx contains TAxis bin indexes
  // here bin count starts at 0 because we do not have over/underflow bins

  Long64_t bin = 0;
  for (Int_t i = 0; i < mNVars; i++) {
    bin *= mPrototype->GetAxis(i)->GetNbins();
    bin += binIdx[i] - 1;
  }

  return bin;
}

void StepTHn::createTarget(Int_t step, Bool_t sparse)
{
  // fills the information stored in the buffer in this class into the target THn

  if (!mValues[step]) {
    LOGF(fatal, "Histogram request for step %d which is empty.", step);
    return;
  }

  if (!mTarget) {
    mTarget = new THnBase*[mNSteps];
    for (Int_t i = 0; i < mNSteps; i++) {
      mTarget[i] = nullptr;
    }
  }

  if (mTarget[step]) {
    return;
  }

  TArray* source = mValues[step];
  // if mSumw2 is not stored, the sqrt of the number of bin entries in source is filled below; otherwise we use mSumw2
  TArray* sourceSumw2 = source;
  if (mSumw2[step]) {
    sourceSumw2 = mSumw2[step];
  }

  if (sparse) {
    mTarget[step] = THnSparse::CreateSparse(Form("%s_%d", GetName(), step), Form("%s_%d", GetTitle(), step), mPrototype);
  } else {
    mTarget[step] = THn::CreateHn(Form("%s_%d", GetName(), step), Form("%s_%d", GetTitle(), step), mPrototype);
  }

  THnBase* target = mTarget[step];

  Int_t* binIdx = new Int_t[mNVars];
  Int_t* nBins = new Int_t[mNVars];
  for (Int_t j = 0; j < mNVars; j++) {
    binIdx[j] = 1;
    nBins[j] = target->GetAxis(j)->GetNbins();
  }

  Long64_t count = 0;

  while (1) {
    //       for (Int_t j=0; j<mNVars; j++)
    //         printf("%d ", binIdx[j]);

    Long64_t globalBin = getGlobalBinIndex(binIdx);
    //       Printf(" --> %lld", globalBin);

    // TODO probably slow
    double value = source->GetAt(globalBin);
    if (value != 0) {
      target->SetBinContent(binIdx, value);
      target->SetBinError(binIdx, TMath::Sqrt(sourceSumw2->GetAt(globalBin)));

      count++;
    }

    binIdx[mNVars - 1]++;

    for (Int_t j = mNVars - 1; j > 0; j--) {
      if (binIdx[j] > nBins[j]) {
        binIdx[j] = 1;
        binIdx[j - 1]++;
      }
    }

    if (binIdx[0] > nBins[0]) {
      break;
    }
  }

  LOGF(info, "Step %d: copied %lld entries out of %lld bins", step, count, getGlobalBinIndex(binIdx));

  delete[] binIdx;
  delete[] nBins;

  delete mValues[step];
  mValues[step] = nullptr;
}

void StepTHn::Fill(int iStep, int nParams, double positionAndWeight[])
{
  if (iStep >= mNSteps) {
    LOGF(fatal, "Selected step for filling is not in range of StepTHn.");
  }

  double weight = 1.0;
  if (nParams == mNVars + 1) {
    weight = positionAndWeight[mNVars];
  } else if (nParams != mNVars) {
    LOGF(fatal, "Fill called with invalid number of parameters (%d vs %d)", mNVars, nParams);
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
      mLastVars[i] = positionAndWeight[i];
      mLastBins[i] = mAxisCache[i]->FindBin(mLastVars[i]);
    }
  }

  // calculate global bin index
  Long64_t bin = 0;
  for (Int_t i = 0; i < mNVars; i++) {
    bin *= mNbinsCache[i];

    Int_t tmpBin = 0;
    if (mLastVars[i] == positionAndWeight[i]) {
      tmpBin = mLastBins[i];
    } else {
      tmpBin = mAxisCache[i]->FindBin(positionAndWeight[i]);
      mLastBins[i] = tmpBin;
      mLastVars[i] = positionAndWeight[i];
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

  if (!mValues[iStep]) {
    mValues[iStep] = createArray();
    LOGF(info, "Created values container for step %d", iStep);
  }

  if (weight != 1.) {
    // initialize with already filled entries (which have been filled with weight == 1), in this case mSumw2 := mValues
    if (!mSumw2[iStep]) {
      mSumw2[iStep] = createArray();
      LOGF(info, "Created sumw2 container for step %d", iStep);
    }
  }

  // TODO probably slow; add StepTHnT::add ?
  mValues[iStep]->SetAt(mValues[iStep]->GetAt(bin) + weight, bin);
  if (mSumw2[iStep]) {
    mSumw2[iStep]->SetAt(mSumw2[iStep]->GetAt(bin) + weight, bin);
  }
}

template class StepTHnT<TArrayF>;
template class StepTHnT<TArrayD>;
