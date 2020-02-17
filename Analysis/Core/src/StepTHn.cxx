// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Analysis/StepTHn.h"
#include "TList.h"
#include "TCollection.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "THn.h"
#include "TMath.h"

// for LOGF
#include "Framework/AnalysisTask.h"

templateClassImp(StepTHn)

  template <class TemplateArray, typename TemplateType>
  StepTHn<TemplateArray, TemplateType>::StepTHn() : StepTHnBase(),
                                                    mNBins(0),
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
  // Constructor
}

template <class TemplateArray, typename TemplateType>
StepTHn<TemplateArray, TemplateType>::StepTHn(const Char_t* name, const Char_t* title, const Int_t nSteps, const Int_t nAxis, Int_t* nBins, Double_t** binEdges, const char** axisTitles) : StepTHnBase(name, title, nSteps, nAxis, nBins, binEdges, axisTitles),
                                                                                                                                                                                            mNBins(0),
                                                                                                                                                                                            mNVars(nAxis),
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

  mNBins = 1;
  for (Int_t i = 0; i < mNVars; i++)
    mNBins *= nBins[i];

  // TODO this should be double for StepTHnD

  mPrototype = new THnSparseF(Form("%s_sparse", name), title, nAxis, nBins);
  for (Int_t i = 0; i < mNVars; i++) {
    mPrototype->SetBinEdges(i, binEdges[i]);
    mPrototype->GetAxis(i)->SetTitle(axisTitles[i]);
  }

  init();
}

template <class TemplateArray, typename TemplateType>
void StepTHn<TemplateArray, TemplateType>::StepTHn::init()
{
  // initialize

  mValues = new TemplateArray*[mNSteps];
  mSumw2 = new TemplateArray*[mNSteps];

  for (Int_t i = 0; i < mNSteps; i++) {
    mValues[i] = nullptr;
    mSumw2[i] = nullptr;
  }
}

template <class TemplateArray, typename TemplateType>
StepTHn<TemplateArray, TemplateType>::StepTHn(const StepTHn<TemplateArray, TemplateType>& c) : StepTHnBase(c),
                                                                                               mNBins(c.mNBins),
                                                                                               mNVars(c.mNVars),
                                                                                               mNSteps(c.mNSteps),
                                                                                               mValues(new TemplateArray*[c.mNSteps]),
                                                                                               mSumw2(new TemplateArray*[c.mNSteps]),
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

template <class TemplateArray, typename TemplateType>
StepTHn<TemplateArray, TemplateType>::~StepTHn()
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

template <class TemplateArray, typename TemplateType>
void StepTHn<TemplateArray, TemplateType>::deleteContainers()
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

//____________________________________________________________________
template <class TemplateArray, typename TemplateType>
StepTHn<TemplateArray, TemplateType>& StepTHn<TemplateArray, TemplateType>::operator=(const StepTHn<TemplateArray, TemplateType>& c)
{
  // assigment operator

  if (this != &c)
    ((StepTHn&)c).Copy(*this);

  return *this;
}

//____________________________________________________________________
template <class TemplateArray, typename TemplateType>
void StepTHn<TemplateArray, TemplateType>::Copy(TObject& c) const
{
  // copy function

  StepTHn& target = (StepTHn&)c;

  TNamed::Copy(c);

  target.mNBins = mNBins;
  target.mNVars = mNVars;
  target.mNSteps = mNSteps;

  target.init();

  for (Int_t i = 0; i < mNSteps; i++) {
    if (mValues[i])
      target.mValues[i] = new TemplateArray(*(mValues[i]));
    else
      target.mValues[i] = nullptr;

    if (mSumw2[i])
      target.mSumw2[i] = new TemplateArray(*(mSumw2[i]));
    else
      target.mSumw2[i] = nullptr;
  }

  if (mPrototype)
    target.mPrototype = dynamic_cast<THnSparseF*>(mPrototype->Clone());
}

//____________________________________________________________________
template <class TemplateArray, typename TemplateType>
Long64_t StepTHn<TemplateArray, TemplateType>::Merge(TCollection* list)
{
  // Merge a list of StepTHn objects with this (needed for
  // PROOF).
  // Returns the number of merged objects (including this).

  if (!list)
    return 0;

  if (list->IsEmpty())
    return 1;

  TIterator* iter = list->MakeIterator();
  TObject* obj;

  Int_t count = 0;
  while ((obj = iter->Next())) {

    StepTHn* entry = dynamic_cast<StepTHn*>(obj);
    if (entry == nullptr)
      continue;

    for (Int_t i = 0; i < mNSteps; i++) {
      if (entry->mValues[i]) {
        if (!mValues[i])
          mValues[i] = new TemplateArray(mNBins);

        for (Long64_t l = 0; l < mNBins; l++)
          mValues[i]->GetArray()[l] += entry->mValues[i]->GetArray()[l];
      }

      if (entry->mSumw2[i]) {
        if (!mSumw2[i])
          mSumw2[i] = new TemplateArray(mNBins);

        for (Long64_t l = 0; l < mNBins; l++)
          mSumw2[i]->GetArray()[l] += entry->mSumw2[i]->GetArray()[l];
      }
    }

    count++;
  }

  return count + 1;
}

template <class TemplateArray, typename TemplateType>
void StepTHn<TemplateArray, TemplateType>::Fill(const Double_t* var, Int_t istep, Double_t weight)
{
  // fills an entry

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
      mLastBins[i] = mAxisCache[i]->FindBin(var[i]);
      mLastVars[i] = var[i];
    }
  }

  // calculate global bin index
  Long64_t bin = 0;
  for (Int_t i = 0; i < mNVars; i++) {
    bin *= mNbinsCache[i];

    Int_t tmpBin = 0;
    if (mLastVars[i] == var[i])
      tmpBin = mLastBins[i];
    else {
      tmpBin = mAxisCache[i]->FindBin(var[i]);
      mLastBins[i] = tmpBin;
      mLastVars[i] = var[i];
    }
    //Printf("%d", tmpBin);

    // under/overflow not supported
    if (tmpBin < 1 || tmpBin > mNbinsCache[i])
      return;

    // bins start from 0 here
    bin += tmpBin - 1;
    //     Printf("%lld", bin);
  }

  if (!mValues[istep]) {
    mValues[istep] = new TemplateArray(mNBins);
    LOGF(info, "Created values container for step %d", istep);
  }

  if (weight != 1) {
    // initialize with already filled entries (which have been filled with weight == 1), in this case mSumw2 := mValues
    if (!mSumw2[istep]) {
      mSumw2[istep] = new TemplateArray(*mValues[istep]);
      LOGF(info, "Created sumw2 container for step %d", istep);
    }
  }

  mValues[istep]->GetArray()[bin] += weight;
  if (mSumw2[istep])
    mSumw2[istep]->GetArray()[bin] += weight * weight;

  //   Printf("%f", mValues[istep][bin]);

  // debug
  //   AliCFContainer::Fill(var, istep, weight);
}

template <class TemplateArray, typename TemplateType>
Long64_t StepTHn<TemplateArray, TemplateType>::getGlobalBinIndex(const Int_t* binIdx)
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

template <class TemplateArray, typename TemplateType>
void StepTHn<TemplateArray, TemplateType>::createTarget(Int_t step, Bool_t sparse)
{
  // fills the information stored in the buffer in this class into the target THn

  if (!mValues[step]) {
    LOGF(fatal, "Histogram request for step %d which is empty.", step);
    return;
  }

  if (!mTarget) {
    mTarget = new THnBase*[mNSteps];
    for (Int_t i = 0; i < mNSteps; i++)
      mTarget[i] = nullptr;
  }

  if (mTarget[step])
    return;

  TemplateType* source = mValues[step]->GetArray();
  // if mSumw2 is not stored, the sqrt of the number of bin entries in source is filled below; otherwise we use mSumw2
  TemplateType* sourceSumw2 = source;
  if (mSumw2[step])
    sourceSumw2 = mSumw2[step]->GetArray();

  if (sparse)
    mTarget[step] = THnSparse::CreateSparse(Form("%s_%d", GetName(), step), Form("%s_%d", GetTitle(), step), mPrototype);
  else
    mTarget[step] = THn::CreateHn(Form("%s_%d", GetName(), step), Form("%s_%d", GetTitle(), step), mPrototype);

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

    if (source[globalBin] != 0) {
      target->SetBinContent(binIdx, source[globalBin]);
      target->SetBinError(binIdx, TMath::Sqrt(sourceSumw2[globalBin]));

      count++;
    }

    binIdx[mNVars - 1]++;

    for (Int_t j = mNVars - 1; j > 0; j--) {
      if (binIdx[j] > nBins[j]) {
        binIdx[j] = 1;
        binIdx[j - 1]++;
      }
    }

    if (binIdx[0] > nBins[0])
      break;
  }

  LOGF(info, "Step %d: copied %lld entries out of %lld bins", step, count, getGlobalBinIndex(binIdx));

  delete[] binIdx;
  delete[] nBins;

  delete mValues[step];
  mValues[step] = nullptr;
}

/*
template <class TemplateArray, typename TemplateType>
void StepTHn<TemplateArray, TemplateType>::FillParent()
{
  // fills the information stored in the buffer in this class into the baseclass containers
  
  FillContainer(this);
}

template <class TemplateArray, typename TemplateType>
void StepTHn<TemplateArray, TemplateType>::ReduceAxis()
{
  // "removes" one axis by summing over the axis and putting the entry to bin 1
  // TODO presently only implemented for the last axis
  
  Int_t axis = mNVars-1;
  
  for (Int_t i=0; i<mNSteps; i++)
  {
    if (!mValues[i])
      continue;
      
    TemplateType* source = mValues[i]->GetArray();
    TemplateType* sourceSumw2 = 0;
    if (mSumw2[i])
      sourceSumw2 = mSumw2[i]->GetArray();
    
    THnSparse* target = GetGrid(i)->GetGrid();
    
    Int_t* binIdx = new Int_t[mNVars];
    Int_t* nBins  = new Int_t[mNVars];
    for (Int_t j=0; j<mNVars; j++)
    {
      binIdx[j] = 1;
      nBins[j] = target->GetAxis(j)->GetNbins();
    }
    
    Long64_t count = 0;

    while (1)
    {
      // sum over axis <axis>
      TemplateType sumValues = 0;
      TemplateType sumSumw2 = 0;
      for (Int_t j=1; j<=nBins[axis]; j++)
      {
        binIdx[axis] = j;
        Long64_t globalBin = getGlobalBinIndex(binIdx);
        sumValues += source[globalBin];
        source[globalBin] = 0;

        if (sourceSumw2)
        {
          sumSumw2 += sourceSumw2[globalBin];
          sourceSumw2[globalBin] = 0;
        }
      }
      binIdx[axis] = 1;
        
      Long64_t globalBin = getGlobalBinIndex(binIdx);
      source[globalBin] = sumValues;
      if (sourceSumw2)
        sourceSumw2[globalBin] = sumSumw2;

      count++;

      // next bin
      binIdx[mNVars-2]++;
      
      for (Int_t j=mNVars-2; j>0; j--)
      {
        if (binIdx[j] > nBins[j])
        {
          binIdx[j] = 1;
          binIdx[j-1]++;
        }
      }
      
      if (binIdx[0] > nBins[0])
        break;
    }
    
    AliInfo(Form("Step %d: reduced %lld bins to %lld entries", i, getGlobalBinIndex(binIdx), count));

    delete[] binIdx;
    delete[] nBins;
  }
}*/

template class StepTHn<TArrayF, Float_t>;
template class StepTHn<TArrayD, Double_t>;
