// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
// encapsulate histogram and corrections for correlation analysis
//
// Author: Jan Fiete Grosse-Oetringhaus

#include "AnalysisCore/CorrelationContainer.h"
#include "Framework/StepTHn.h"
#include "Framework/Logger.h"
#include "THnSparse.h"
#include "TMath.h"
#include "TList.h"
#include "TCollection.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TCanvas.h"
#include "TF1.h"
#include "THn.h"
#include "Framework/HistogramSpec.h"

using namespace o2::framework;

ClassImp(CorrelationContainer);

const Int_t CorrelationContainer::fgkCFSteps = 11;

CorrelationContainer::CorrelationContainer() : TNamed(),
                                               mPairHist(nullptr),
                                               mTriggerHist(nullptr),
                                               mTrackHistEfficiency(nullptr),
                                               mEventCount(nullptr),
                                               mEtaMin(0),
                                               mEtaMax(0),
                                               mPtMin(0),
                                               mPtMax(0),
                                               mPartSpecies(-1),
                                               mCentralityMin(0),
                                               mCentralityMax(0),
                                               mZVtxMin(0),
                                               mZVtxMax(0),
                                               mPt2Min(0),
                                               mPt2Max(0),
                                               mTrackEtaCut(0),
                                               mWeightPerEvent(0),
                                               mSkipScaleMixedEvent(kFALSE),
                                               mCache(nullptr),
                                               mGetMultCacheOn(kFALSE),
                                               mGetMultCache(nullptr)
{
  // Default constructor
}

CorrelationContainer::CorrelationContainer(const char* name, const char* objTitle, const std::vector<o2::framework::AxisSpec>& axisList) : TNamed(name, objTitle),
                                                                                                                                           mPairHist(nullptr),
                                                                                                                                           mTriggerHist(nullptr),
                                                                                                                                           mTrackHistEfficiency(nullptr),
                                                                                                                                           mEventCount(nullptr),
                                                                                                                                           mEtaMin(0),
                                                                                                                                           mEtaMax(0),
                                                                                                                                           mPtMin(0),
                                                                                                                                           mPtMax(0),
                                                                                                                                           mPartSpecies(-1),
                                                                                                                                           mCentralityMin(0),
                                                                                                                                           mCentralityMax(0),
                                                                                                                                           mZVtxMin(0),
                                                                                                                                           mZVtxMax(0),
                                                                                                                                           mPt2Min(0),
                                                                                                                                           mPt2Max(0),
                                                                                                                                           mTrackEtaCut(0),
                                                                                                                                           mWeightPerEvent(0),
                                                                                                                                           mSkipScaleMixedEvent(kFALSE),
                                                                                                                                           mCache(nullptr),
                                                                                                                                           mGetMultCacheOn(kFALSE),
                                                                                                                                           mGetMultCache(nullptr)
{
  // Constructor
  //
  // axisList has to provide a 8 length list of AxisSpec which contain:
  //    delta_eta, pt_assoc, pt_trig, multiplicity/centrality, delta_phi, vertex, eta (for efficiency), pt (for efficiency), vertex (for efficiency)

  if (strlen(name) == 0) {
    return;
  }

  LOGF(info, "Creating CorrelationContainer");

  mPairHist = HistFactory::createHist<StepTHnF>({"mPairHist", "d^{2}N_{ch}/d#varphid#eta", {HistType::kStepTHnF, {axisList[0], axisList[1], axisList[2], axisList[3], axisList[4], axisList[5]}, fgkCFSteps}}).release();
  mTriggerHist = HistFactory::createHist<StepTHnF>({"mTriggerHist", "d^{2}N_{ch}/d#varphid#eta", {HistType::kStepTHnF, {axisList[2], axisList[3], axisList[5]}, fgkCFSteps}}).release();
  mTrackHistEfficiency = HistFactory::createHist<StepTHnD>({"mTrackHistEfficiency", "Tracking efficiency", {HistType::kStepTHnD, {axisList[6], axisList[7], {4, -0.5, 3.5, "species"}, axisList[3], axisList[8]}, fgkCFSteps}}).release();
  mEventCount = HistFactory::createHist<TH2F>({"mEventCount", ";step;centrality;count", {HistType::kTH2F, {{fgkCFSteps + 2, -2.5, -0.5 + fgkCFSteps, "step"}, axisList[3]}}}).release();
}

//_____________________________________________________________________________
CorrelationContainer::CorrelationContainer(const CorrelationContainer& c) : TNamed(c),
                                                                            mPairHist(nullptr),
                                                                            mTriggerHist(nullptr),
                                                                            mTrackHistEfficiency(nullptr),
                                                                            mEtaMin(0),
                                                                            mEtaMax(0),
                                                                            mPtMin(0),
                                                                            mPtMax(0),
                                                                            mPartSpecies(-1),
                                                                            mCentralityMin(0),
                                                                            mCentralityMax(0),
                                                                            mZVtxMin(0),
                                                                            mZVtxMax(0),
                                                                            mPt2Min(0),
                                                                            mPt2Max(0),
                                                                            mTrackEtaCut(0),
                                                                            mWeightPerEvent(0),
                                                                            mSkipScaleMixedEvent(kFALSE),
                                                                            mCache(nullptr),
                                                                            mGetMultCacheOn(kFALSE),
                                                                            mGetMultCache(nullptr)
{
  //
  // CorrelationContainer copy constructor
  //

  ((CorrelationContainer&)c).Copy(*this);
}

//____________________________________________________________________
CorrelationContainer::~CorrelationContainer()
{
  // Destructor

  if (mPairHist) {
    delete mPairHist;
    mPairHist = nullptr;
  }

  if (mTriggerHist) {
    delete mTriggerHist;
    mTriggerHist = nullptr;
  }

  if (mTrackHistEfficiency) {
    delete mTrackHistEfficiency;
    mTrackHistEfficiency = nullptr;
  }

  if (mEventCount) {
    delete mEventCount;
    mEventCount = nullptr;
  }

  if (mCache) {
    delete mCache;
    mCache = nullptr;
  }
}

//____________________________________________________________________
CorrelationContainer& CorrelationContainer::operator=(const CorrelationContainer& c)
{
  // assigment operator

  if (this != &c) {
    ((CorrelationContainer&)c).Copy(*this);
  }

  return *this;
}

//____________________________________________________________________
void CorrelationContainer::Copy(TObject& c) const
{
  // copy function

  CorrelationContainer& target = (CorrelationContainer&)c;

  if (mPairHist) {
    target.mPairHist = dynamic_cast<StepTHn*>(mPairHist->Clone());
  }

  if (mTriggerHist) {
    target.mTriggerHist = dynamic_cast<StepTHn*>(mTriggerHist->Clone());
  }

  if (mTrackHistEfficiency) {
    target.mTrackHistEfficiency = dynamic_cast<StepTHn*>(mTrackHistEfficiency->Clone());
  }

  target.mEtaMin = mEtaMin;
  target.mEtaMax = mEtaMax;
  target.mPtMin = mPtMin;
  target.mPtMax = mPtMax;
  target.mPartSpecies = mPartSpecies;
  target.mCentralityMin = mCentralityMin;
  target.mCentralityMax = mCentralityMax;
  target.mZVtxMin = mZVtxMin;
  target.mZVtxMax = mZVtxMax;
  target.mPt2Min = mPt2Min;
  target.mPt2Max = mPt2Max;
  target.mTrackEtaCut = mTrackEtaCut;
  target.mWeightPerEvent = mWeightPerEvent;
  target.mSkipScaleMixedEvent = mSkipScaleMixedEvent;
}

//____________________________________________________________________
Long64_t CorrelationContainer::Merge(TCollection* list)
{
  // Merge a list of CorrelationContainer objects with this (needed for
  // PROOF).
  // Returns the number of merged objects (including this).

  if (!list) {
    return 0;
  }

  if (list->IsEmpty()) {
    return 1;
  }

  TIterator* iter = list->MakeIterator();
  TObject* obj = nullptr;

  // collections of objects
  const UInt_t kMaxLists = 4;
  TList** lists = new TList*[kMaxLists];

  for (UInt_t i = 0; i < kMaxLists; i++) {
    lists[i] = new TList;
  }

  Int_t count = 0;
  while ((obj = iter->Next())) {

    CorrelationContainer* entry = dynamic_cast<CorrelationContainer*>(obj);
    if (entry == nullptr) {
      continue;
    }

    if (entry->mPairHist) {
      lists[0]->Add(entry->mPairHist);
    }

    lists[1]->Add(entry->mTriggerHist);
    lists[2]->Add(entry->mTrackHistEfficiency);
    lists[3]->Add(entry->mEventCount);

    count++;
  }
  if (mPairHist) {
    mPairHist->Merge(lists[0]);
  }

  mTriggerHist->Merge(lists[1]);
  mTrackHistEfficiency->Merge(lists[2]);
  mEventCount->Merge(lists[3]);

  for (UInt_t i = 0; i < kMaxLists; i++) {
    delete lists[i];
  }

  delete[] lists;

  return count + 1;
}

//____________________________________________________________________
void CorrelationContainer::setBinLimits(THnBase* grid)
{
  // sets the bin limits in eta and pT defined by mEtaMin/Max, mPtMin/Max

  if (mEtaMax > mEtaMin) {
    grid->GetAxis(0)->SetRangeUser(mEtaMin, mEtaMax);
  }
  if (mPtMax > mPtMin) {
    grid->GetAxis(1)->SetRangeUser(mPtMin, mPtMax);
  }
  if (mPt2Min > 0 && mPt2Max > 0) {
    grid->GetAxis(6)->SetRangeUser(mPt2Min, mPt2Max);
  } else if (mPt2Min > 0) {
    grid->GetAxis(6)->SetRangeUser(mPt2Min, grid->GetAxis(6)->GetXmax() - 0.01);
  }
}

//____________________________________________________________________
void CorrelationContainer::resetBinLimits(THnBase* grid)
{
  // resets all bin limits

  for (Int_t i = 0; i < grid->GetNdimensions(); i++) {
    if (grid->GetAxis(i)->TestBit(TAxis::kAxisRange)) {
      grid->GetAxis(i)->SetRangeUser(0, -1);
    }
  }
}

//____________________________________________________________________
void CorrelationContainer::countEmptyBins(CorrelationContainer::CFStep step, Float_t ptTriggerMin, Float_t ptTriggerMax)
{
  // prints the number of empty bins in the track end event histograms in the given step

  Int_t binBegin[4];
  Int_t binEnd[4];

  for (Int_t i = 0; i < 4; i++) {
    binBegin[i] = 1;
    binEnd[i] = mPairHist->getTHn(step)->GetAxis(i)->GetNbins();
  }

  if (mEtaMax > mEtaMin) {
    binBegin[0] = mPairHist->getTHn(step)->GetAxis(0)->FindBin(mEtaMin);
    binEnd[0] = mPairHist->getTHn(step)->GetAxis(0)->FindBin(mEtaMax);
  }

  if (mPtMax > mPtMin) {
    binBegin[1] = mPairHist->getTHn(step)->GetAxis(1)->FindBin(mPtMin);
    binEnd[1] = mPairHist->getTHn(step)->GetAxis(1)->FindBin(mPtMax);
  }

  if (ptTriggerMax > ptTriggerMin) {
    binBegin[2] = mPairHist->getTHn(step)->GetAxis(2)->FindBin(ptTriggerMin);
    binEnd[2] = mPairHist->getTHn(step)->GetAxis(2)->FindBin(ptTriggerMax);
  }

  // start from multiplicity 1
  binBegin[3] = mPairHist->getTHn(step)->GetAxis(3)->FindBin(1);

  Int_t total = 0;
  Int_t count = 0;
  Int_t vars[4];

  for (Int_t i = 0; i < 4; i++) {
    vars[i] = binBegin[i];
  }

  THnBase* grid = mPairHist->getTHn(step);
  while (1) {
    if (grid->GetBin(vars) == 0) {
      LOGF(warning, "Empty bin at eta=%.2f pt=%.2f pt_lead=%.2f mult=%.1f",
           grid->GetAxis(0)->GetBinCenter(vars[0]),
           grid->GetAxis(1)->GetBinCenter(vars[1]),
           grid->GetAxis(2)->GetBinCenter(vars[2]),
           grid->GetAxis(3)->GetBinCenter(vars[3]));
      count++;
    }

    vars[3]++;
    for (Int_t i = 3; i > 0; i--) {
      if (vars[i] == binEnd[i] + 1) {
        vars[i] = binBegin[i];
        vars[i - 1]++;
      }
    }

    if (vars[0] == binEnd[0] + 1) {
      break;
    }
    total++;
  }

  LOGF(info, "Has %d empty bins (out of %d bins)", count, total);
}

//____________________________________________________________________
void CorrelationContainer::getHistsZVtxMult(CorrelationContainer::CFStep step, Float_t ptTriggerMin, Float_t ptTriggerMax, THnBase** trackHist, TH2** eventHist)
{
  // Calculates a 4d histogram with deltaphi, deltaeta, zvtx, multiplicity on track level and
  // a 2d histogram on event level (as fct of zvtx, multiplicity)
  // Histograms has to be deleted by the caller of the function

  THnBase* sparse = mPairHist->getTHn(step);
  if (mGetMultCacheOn) {
    if (!mGetMultCache) {
      mGetMultCache = changeToThn(sparse);
      // should work but causes SEGV in ProjectionND below
    }
    sparse = mGetMultCache;
  }

  // unzoom all axes
  resetBinLimits(sparse);
  resetBinLimits(mTriggerHist->getTHn(step));

  setBinLimits(sparse);

  Int_t firstBin = sparse->GetAxis(2)->FindBin(ptTriggerMin);
  Int_t lastBin = sparse->GetAxis(2)->FindBin(ptTriggerMax);
  LOGF(info, "Using trigger pT range %d --> %d", firstBin, lastBin);
  sparse->GetAxis(2)->SetRange(firstBin, lastBin);
  mTriggerHist->getTHn(step)->GetAxis(0)->SetRange(firstBin, lastBin);

  // cut on the second trigger particle if there is a minimum set
  if (mPt2Min > 0) {
    Int_t firstBinPt2 = sparse->GetAxis(6)->FindBin(mPt2Min);
    Int_t lastBinPt2 = sparse->GetAxis(6)->GetNbins();
    if (mPt2Max > 0) {
      lastBinPt2 = sparse->GetAxis(6)->FindBin(mPt2Max);
    }

    mTriggerHist->getTHn(step)->GetAxis(3)->SetRange(firstBinPt2, lastBinPt2);
  }

  Bool_t hasVertex = kTRUE;
  if (!mPairHist->getTHn(step)->GetAxis(5)) {
    hasVertex = kFALSE;
  }

  if (hasVertex) {
    Int_t dimensions[] = {4, 0, 5, 3};
    THnBase* tmpTrackHist = sparse->ProjectionND(4, dimensions, "E");
    *eventHist = (TH2*)mTriggerHist->getTHn(step)->Projection(2, 1);
    // convert to THn
    *trackHist = changeToThn(tmpTrackHist);
    delete tmpTrackHist;
  } else {
    Int_t dimensions[] = {4, 0, 3};
    THnBase* tmpTrackHist = sparse->ProjectionND(3, dimensions, "E");

    // add dummy vertex axis, so that the extraction code can work as usual
    Int_t nBins[] = {tmpTrackHist->GetAxis(0)->GetNbins(), tmpTrackHist->GetAxis(1)->GetNbins(), 1, tmpTrackHist->GetAxis(2)->GetNbins()};
    Double_t vtxAxis[] = {-100, 100};

    *trackHist = new THnF(Form("%s_thn", tmpTrackHist->GetName()), tmpTrackHist->GetTitle(), 4, nBins, nullptr, nullptr);

    for (int i = 0; i < 3; i++) {
      int j = i;
      if (i == 2) {
        j = 3;
      }

      (*trackHist)->SetBinEdges(j, tmpTrackHist->GetAxis(i)->GetXbins()->GetArray());
      (*trackHist)->GetAxis(j)->SetTitle(tmpTrackHist->GetAxis(i)->GetTitle());
    }

    (*trackHist)->SetBinEdges(2, vtxAxis);
    (*trackHist)->GetAxis(2)->SetTitle("dummy z-vtx");

    // bin by bin copy...
    Int_t bins[4];
    for (Int_t binIdx = 0; binIdx < tmpTrackHist->GetNbins(); binIdx++) {
      Double_t value = tmpTrackHist->GetBinContent(binIdx, bins);
      Double_t error = tmpTrackHist->GetBinError(binIdx);

      // move third to fourth axis
      bins[3] = bins[2];
      bins[2] = 1;

      (*trackHist)->SetBinContent(bins, value);
      (*trackHist)->SetBinError(bins, error);
    }

    delete tmpTrackHist;

    TH1* projEventHist = (TH1*)mTriggerHist->getTHn(step)->Projection(1);
    *eventHist = new TH2F(Form("%s_vtx", projEventHist->GetName()), projEventHist->GetTitle(), 1, vtxAxis, projEventHist->GetNbinsX(), projEventHist->GetXaxis()->GetXbins()->GetArray());
    for (Int_t binIdx = 1; binIdx <= projEventHist->GetNbinsX(); binIdx++) {
      (*eventHist)->SetBinContent(1, binIdx, projEventHist->GetBinContent(binIdx));
      (*eventHist)->SetBinError(1, binIdx, projEventHist->GetBinError(binIdx));
    }

    delete projEventHist;
  }

  resetBinLimits(sparse);
  resetBinLimits(mTriggerHist->getTHn(step));
}

TH2* CorrelationContainer::getPerTriggerYield(CorrelationContainer::CFStep step, Float_t ptTriggerMin, Float_t ptTriggerMax, Bool_t normalizePerTrigger)
{
  // Calculate per trigger yield without considering mixed event
  // Sums over all vertex bins, and respects the pT and centrality limits set in the class
  //
  // returns a 2D histogram: deltaphi, deltaeta
  //
  // Parameters:
  //   step: Step for which the histogram is received
  //   ptTriggerMin, ptTriggerMax: trigger pT range
  //   normalizePerTrigger: divide through number of triggers

  THnBase* trackSameAll = nullptr;
  TH2* eventSameAll = nullptr;

  Long64_t totalEvents = 0;
  Int_t nCorrelationFunctions = 0;

  getHistsZVtxMult(step, ptTriggerMin, ptTriggerMax, &trackSameAll, &eventSameAll);

  TAxis* multAxis = trackSameAll->GetAxis(3);
  int multBinBegin = 1;
  int multBinEnd = multAxis->GetNbins();
  if (mCentralityMin < mCentralityMax) {
    multBinBegin = multAxis->FindBin(mCentralityMin + 1e-4);
    multBinEnd = multAxis->FindBin(mCentralityMax - 1e-4);
    LOGF(info, "Using multiplicity range %d --> %d", multBinBegin, multBinEnd);

    trackSameAll->GetAxis(3)->SetRange(multBinBegin, multBinEnd);
  }

  TAxis* vertexAxis = trackSameAll->GetAxis(2);
  int vertexBinBegin = 1;
  int vertexBinEnd = vertexAxis->GetNbins();
  if (mZVtxMax > mZVtxMin) {
    vertexBinBegin = vertexAxis->FindBin(mZVtxMin);
    vertexBinEnd = vertexAxis->FindBin(mZVtxMax);
    LOGF(info, "Using vertex range %d --> %d", vertexBinBegin, vertexBinEnd);

    trackSameAll->GetAxis(2)->SetRange(vertexBinBegin, vertexBinEnd);
  }

  TH2* yield = trackSameAll->Projection(1, 0, "E");
  Float_t triggers = eventSameAll->Integral(multBinBegin, multBinEnd, vertexBinBegin, vertexBinEnd);

  if (normalizePerTrigger) {
    LOGF(info, "Dividing %f tracks by %f triggers", yield->Integral(), triggers);
    if (triggers > 0) {
      yield->Scale(1.0 / triggers);
    }
  }

  // normalizate to dphi width
  Float_t normalization = yield->GetXaxis()->GetBinWidth(1);
  yield->Scale(1.0 / normalization);

  delete trackSameAll;
  delete eventSameAll;

  return yield;
}

TH2* CorrelationContainer::getSumOfRatios(CorrelationContainer* mixed, CorrelationContainer::CFStep step, Float_t ptTriggerMin, Float_t ptTriggerMax, Bool_t normalizePerTrigger, Int_t stepForMixed, Int_t* trigger)
{
  // Extract 2D per trigger yield with mixed event correction. The quantity is calculated for *each* vertex bin and multiplicity bin and then a sum of ratios is performed:
  // 1_N [ (same/mixed)_1 + (same/mixed)_2 + (same/mixed)_3 + ... ]
  // where N is the total number of events/trigger particles and the subscript is the vertex/multiplicity bin
  // where mixed is normalized such that the information about the number of pairs in same is kept
  //
  // returns a 2D histogram: deltaphi, deltaeta
  //
  // Parameters:
  //   mixed: CorrelationContainer containing mixed event corresponding to this object (the histograms are taken from step <stepForMixed> if defined otherwise from step <step>)
  //   step: Step for which the histogram is received
  //   ptTriggerMin, ptTriggerMax: trigger pT range
  //   normalizePerTrigger: divide through number of triggers

  // do not add this hists to the directory
  Bool_t oldStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  TH2* totalTracks = nullptr;

  THnBase* trackSameAll = nullptr;
  THnBase* trackMixedAll = nullptr;
  THnBase* trackMixedAllStep6 = nullptr;
  TH2* eventSameAll = nullptr;
  TH2* eventMixedAll = nullptr;
  TH2* eventMixedAllStep6 = nullptr;

  Long64_t totalEvents = 0;
  Int_t nCorrelationFunctions = 0;

  getHistsZVtxMult(step, ptTriggerMin, ptTriggerMax, &trackSameAll, &eventSameAll);
  mixed->getHistsZVtxMult((stepForMixed == -1) ? step : (CFStep)stepForMixed, ptTriggerMin, ptTriggerMax, &trackMixedAll, &eventMixedAll);

  // If we ask for histograms from step8 (TTR cut applied) there is a hole at 0,0; so this cannot be used for the
  // mixed-event normalization. If step6 is available, the normalization factor is read out from that one.
  // If step6 is not available we fallback to taking the normalization along all delta phi (WARNING requires a
  // flat delta phi distribution)
  if (stepForMixed == -1 && step == kCFStepBiasStudy && mixed->mTriggerHist->getTHn(kCFStepReconstructed)->GetEntries() > 0 && !mSkipScaleMixedEvent) {
    LOGF(info, "Using mixed-event normalization factors from step %d", kCFStepReconstructed);
    mixed->getHistsZVtxMult(kCFStepReconstructed, ptTriggerMin, ptTriggerMax, &trackMixedAllStep6, &eventMixedAllStep6);
  }

  //   Printf("%f %f %f %f", trackSameAll->GetEntries(), eventSameAll->GetEntries(), trackMixedAll->GetEntries(), eventMixedAll->GetEntries());

  //   TH1* normParameters = new TH1F("normParameters", "", 100, 0, 2);

  //   trackSameAll->Dump();

  TAxis* multAxis = trackSameAll->GetAxis(3);

  int multBinBegin = 1;
  int multBinEnd = multAxis->GetNbins();
  if (mCentralityMin < mCentralityMax) {
    multBinBegin = multAxis->FindBin(mCentralityMin);
    multBinEnd = multAxis->FindBin(mCentralityMax);
  }

  for (Int_t multBin = TMath::Max(1, multBinBegin); multBin <= TMath::Min(multAxis->GetNbins(), multBinEnd); multBin++) {
    trackSameAll->GetAxis(3)->SetRange(multBin, multBin);
    trackMixedAll->GetAxis(3)->SetRange(multBin, multBin);
    if (trackMixedAllStep6) {
      trackMixedAllStep6->GetAxis(3)->SetRange(multBin, multBin);
    }

    Double_t mixedNorm = 1;
    Double_t mixedNormError = 0;

    if (!mSkipScaleMixedEvent) {
      // get mixed normalization correction factor: is independent of vertex bin if scaled with number of triggers
      TH2* tracksMixed = nullptr;
      if (trackMixedAllStep6) {
        trackMixedAllStep6->GetAxis(2)->SetRange(0, -1);
        tracksMixed = trackMixedAllStep6->Projection(1, 0, "E");
      } else {
        trackMixedAll->GetAxis(2)->SetRange(0, -1);
        tracksMixed = trackMixedAll->Projection(1, 0, "E");
      }
      //     Printf("%f", tracksMixed->Integral());
      Float_t binWidthEta = tracksMixed->GetYaxis()->GetBinWidth(1);

      if (stepForMixed == -1 && step == kCFStepBiasStudy && !trackMixedAllStep6) {
        // get mixed event normalization by assuming full acceptance at deta at 0 (integrate over dphi), excluding (0, 0)
        Float_t phiExclude = 0.41;
        mixedNorm = tracksMixed->IntegralAndError(1, tracksMixed->GetXaxis()->FindBin(-phiExclude) - 1, tracksMixed->GetYaxis()->FindBin(-0.01), tracksMixed->GetYaxis()->FindBin(0.01), mixedNormError);
        Double_t mixedNormError2 = 0;
        Double_t mixedNorm2 = tracksMixed->IntegralAndError(tracksMixed->GetXaxis()->FindBin(phiExclude) + 1, tracksMixed->GetNbinsX(), tracksMixed->GetYaxis()->FindBin(-0.01), tracksMixed->GetYaxis()->FindBin(0.01), mixedNormError2);

        if (mixedNormError == 0 || mixedNormError2 == 0) {
          LOGF(error, "ERROR: Skipping multiplicity %d because mixed event is empty %f %f %f %f", multBin, mixedNorm, mixedNormError, mixedNorm2, mixedNormError2);
          continue;
        }

        Int_t nBinsMixedNorm = (tracksMixed->GetXaxis()->FindBin(-phiExclude) - 1 - 1 + 1) * (tracksMixed->GetYaxis()->FindBin(0.01) - tracksMixed->GetYaxis()->FindBin(-0.01) + 1);
        mixedNorm /= nBinsMixedNorm;
        mixedNormError /= nBinsMixedNorm;

        Int_t nBinsMixedNorm2 = (tracksMixed->GetNbinsX() - tracksMixed->GetXaxis()->FindBin(phiExclude) - 1 + 1) * (tracksMixed->GetYaxis()->FindBin(0.01) - tracksMixed->GetYaxis()->FindBin(-0.01) + 1);
        mixedNorm2 /= nBinsMixedNorm2;
        mixedNormError2 /= nBinsMixedNorm2;

        mixedNorm = mixedNorm / mixedNormError / mixedNormError + mixedNorm2 / mixedNormError2 / mixedNormError2;
        mixedNormError = TMath::Sqrt(1.0 / (1.0 / mixedNormError / mixedNormError + 1.0 / mixedNormError2 / mixedNormError2));
        mixedNorm *= mixedNormError * mixedNormError;
      } else {
        // get mixed event normalization at (0,0)

        // NOTE if variable bin size is used around (0,0) to reduce two-track effect to limited bins, the normalization gets a bit tricky here (finite bin correction and normalization are made only for fixed size bins).
        // The normalization factor has to determined in a bin as large as the normal bin size, as a proxy the bin with index (1, 1) is used
        Float_t binWidthPhi = tracksMixed->GetXaxis()->GetBinWidth(1);

        mixedNorm = tracksMixed->IntegralAndError(tracksMixed->GetXaxis()->FindBin(-binWidthPhi + 1e-4), tracksMixed->GetXaxis()->FindBin(binWidthPhi - 1e-4), tracksMixed->GetYaxis()->FindBin(-binWidthEta + 1e-4), tracksMixed->GetYaxis()->FindBin(binWidthEta - 1e-4), mixedNormError);
        Int_t nBinsMixedNorm = 4; // NOTE this is fixed on purpose, even if binning is made finer around (0,0), this corresponds to the equivalent of four "large" bins around (0,0)
        mixedNorm /= nBinsMixedNorm;
        mixedNormError /= nBinsMixedNorm;

        if (mixedNormError == 0) {
          LOGF(error, "ERROR: Skipping multiplicity %d because mixed event is empty %f %f", multBin, mixedNorm, mixedNormError);
          continue;
        }
      }

      // finite bin correction
      if (mTrackEtaCut > 0) {
        Double_t finiteBinCorrection = -1.0 / (2 * mTrackEtaCut) * binWidthEta / 2 + 1;
        LOGF(info, "Finite bin correction: %f", finiteBinCorrection);
        mixedNorm /= finiteBinCorrection;
        mixedNormError /= finiteBinCorrection;
      } else {
        LOGF(error, "ERROR: mTrackEtaCut not set. Finite bin correction cannot be applied. Continuing anyway...");
      }

      Float_t triggers = eventMixedAll->Integral(1, eventMixedAll->GetNbinsX(), multBin, multBin);
      //     Printf("%f +- %f | %f | %f", mixedNorm, mixedNormError, triggers, mixedNorm / triggers);
      if (triggers <= 0) {
        LOGF(error, "ERROR: Skipping multiplicity %d because mixed event is empty", multBin);
        continue;
      }

      mixedNorm /= triggers;
      mixedNormError /= triggers;

      delete tracksMixed;
    } else {
      LOGF(warning, "WARNING: Skipping mixed-event scaling! mSkipScaleMixedEvent IS set!");
    }

    if (mixedNorm <= 0) {
      LOGF(error, "ERROR: Skipping multiplicity %d because mixed event is empty at (0,0)", multBin);
      continue;
    }

    //     Printf("Norm: %f +- %f", mixedNorm, mixedNormError);

    //     normParameters->Fill(mixedNorm);

    TAxis* vertexAxis = trackSameAll->GetAxis(2);
    Int_t vertexBinBegin = 1;
    Int_t vertexBinEnd = vertexAxis->GetNbins();

    if (mZVtxMax > mZVtxMin) {
      vertexBinBegin = vertexAxis->FindBin(mZVtxMin);
      vertexBinEnd = vertexAxis->FindBin(mZVtxMax);
    }

    for (Int_t vertexBin = vertexBinBegin; vertexBin <= vertexBinEnd; vertexBin++) {
      trackSameAll->GetAxis(2)->SetRange(vertexBin, vertexBin);
      trackMixedAll->GetAxis(2)->SetRange(vertexBin, vertexBin);

      TH2* tracksSame = trackSameAll->Projection(1, 0, "E");
      TH2* tracksMixed = trackMixedAll->Projection(1, 0, "E");

      // asssume flat in dphi, gain in statistics
      //     TH1* histMixedproj = mixedTwoD->ProjectionY();
      //     histMixedproj->Scale(1.0 / mixedTwoD->GetNbinsX());
      //
      //     for (Int_t x=1; x<=mixedTwoD->GetNbinsX(); x++)
      //       for (Int_t y=1; y<=mixedTwoD->GetNbinsY(); y++)
      //       mixedTwoD->SetBinContent(x, y, histMixedproj->GetBinContent(y));

      //       delete histMixedproj;

      Float_t triggers2 = eventMixedAll->Integral(vertexBin, vertexBin, multBin, multBin);
      if (triggers2 <= 0) {
        LOGF(error, "ERROR: Skipping multiplicity %d vertex bin %d because mixed event is empty", multBin, vertexBin);
      } else {
        if (!mSkipScaleMixedEvent) {
          tracksMixed->Scale(1.0 / triggers2 / mixedNorm);
        } else if (tracksMixed->Integral() > 0) {
          tracksMixed->Scale(1.0 / tracksMixed->Integral());
        }
        // tracksSame->Scale(tracksMixed->Integral() / tracksSame->Integral());

        // new TCanvas; tracksSame->DrawClone("SURF1");
        // new TCanvas; tracksMixed->DrawClone("SURF1");

        // some code to judge the relative contribution of the different correlation functions to the overall uncertainty
        Double_t sums[] = {0, 0, 0};
        Double_t errors[] = {0, 0, 0};

        for (Int_t x = 1; x <= tracksSame->GetNbinsX(); x++) {
          for (Int_t y = 1; y <= tracksSame->GetNbinsY(); y++) {
            sums[0] += tracksSame->GetBinContent(x, y);
            errors[0] += tracksSame->GetBinError(x, y);
            sums[1] += tracksMixed->GetBinContent(x, y);
            errors[1] += tracksMixed->GetBinError(x, y);
          }
        }

        tracksSame->Divide(tracksMixed);

        for (Int_t x = 1; x <= tracksSame->GetNbinsX(); x++) {
          for (Int_t y = 1; y <= tracksSame->GetNbinsY(); y++) {
            sums[2] += tracksSame->GetBinContent(x, y);
            errors[2] += tracksSame->GetBinError(x, y);
          }
        }

        for (Int_t x = 0; x < 3; x++) {
          if (sums[x] > 0) {
            errors[x] /= sums[x];
          }
        }

        LOGF(info, "The correlation function %d %d has uncertainties %f %f %f (Ratio S/M %f)", multBin, vertexBin, errors[0], errors[1], errors[2], (errors[1] > 0) ? errors[0] / errors[1] : -1);
        // code to draw contributions
        /*
        TH1* proj = tracksSame->ProjectionX("projx", tracksSame->GetYaxis()->FindBin(-1.59), tracksSame->GetYaxis()->FindBin(1.59));
        proj->SetTitle(Form("Bin %d", vertexBin));
        proj->SetLineColor(vertexBin);
        proj->DrawCopy((vertexBin > 1) ? "SAME" : "");
        */

        if (!totalTracks) {
          totalTracks = (TH2*)tracksSame->Clone("totalTracks");
        } else {
          totalTracks->Add(tracksSame);
        }

        totalEvents += eventSameAll->GetBinContent(vertexBin, multBin);

        //   new TCanvas; tracksMixed->DrawCopy("SURF1");
      }

      delete tracksSame;
      delete tracksMixed;

      nCorrelationFunctions++;
    }
  }

  if (totalTracks) {
    Double_t sums[] = {0, 0, 0};
    Double_t errors[] = {0, 0, 0};

    for (Int_t x = 1; x <= totalTracks->GetNbinsX(); x++) {
      for (Int_t y = 1; y <= totalTracks->GetNbinsY(); y++) {
        sums[0] += totalTracks->GetBinContent(x, y);
        errors[0] += totalTracks->GetBinError(x, y);
      }
    }
    if (sums[0] > 0) {
      errors[0] /= sums[0];
    }

    if (normalizePerTrigger) {
      LOGF(info, "Dividing %f tracks by %lld events (%d correlation function(s)) (error %f)", totalTracks->Integral(), totalEvents, nCorrelationFunctions, errors[0]);
      if (totalEvents > 0) {
        totalTracks->Scale(1.0 / totalEvents);
      }
    }
    if (trigger != nullptr) {
      *trigger = (Int_t)totalEvents;
    }

    // normalizate to dphi width
    Float_t normalization = totalTracks->GetXaxis()->GetBinWidth(1);
    totalTracks->Scale(1.0 / normalization);
  }

  delete trackSameAll;
  delete trackMixedAll;
  delete trackMixedAllStep6;
  delete eventSameAll;
  delete eventMixedAll;
  delete eventMixedAllStep6;

  //   new TCanvas; normParameters->Draw();

  TH1::AddDirectory(oldStatus);

  return totalTracks;
}

TH1* CorrelationContainer::getTriggersAsFunctionOfMultiplicity(CorrelationContainer::CFStep step, Float_t ptTriggerMin, Float_t ptTriggerMax)
{
  // returns the distribution of triggers as function of centrality/multiplicity

  resetBinLimits(mTriggerHist->getTHn(step));

  Int_t firstBin = mTriggerHist->getTHn(step)->GetAxis(0)->FindBin(ptTriggerMin);
  Int_t lastBin = mTriggerHist->getTHn(step)->GetAxis(0)->FindBin(ptTriggerMax);
  LOGF(info, "Using pT range %d --> %d", firstBin, lastBin);
  mTriggerHist->getTHn(step)->GetAxis(0)->SetRange(firstBin, lastBin);

  if (mZVtxMax > mZVtxMin) {
    mTriggerHist->getTHn(step)->GetAxis(2)->SetRangeUser(mZVtxMin, mZVtxMax);
    LOGF(info, "Restricting z-vtx: %f-%f", mZVtxMin, mZVtxMax);
  }

  TH1* eventHist = mTriggerHist->getTHn(step)->Projection(1);

  resetBinLimits(mTriggerHist->getTHn(step));

  return eventHist;
}

THnBase* CorrelationContainer::getTrackEfficiencyND(CFStep step1, CFStep step2)
{
  // creates a track-level efficiency by dividing step2 by step1
  // in all dimensions but the particle species one

  StepTHn* sourceContainer = mTrackHistEfficiency;
  // step offset because we start with kCFStepAnaTopology
  step1 = (CFStep)((Int_t)step1 - (Int_t)kCFStepAnaTopology);
  step2 = (CFStep)((Int_t)step2 - (Int_t)kCFStepAnaTopology);

  resetBinLimits(sourceContainer->getTHn(step1));
  resetBinLimits(sourceContainer->getTHn(step2));

  if (mEtaMax > mEtaMin) {
    LOGF(info, "Restricted eta-range to %f %f", mEtaMin, mEtaMax);
    sourceContainer->getTHn(step1)->GetAxis(0)->SetRangeUser(mEtaMin, mEtaMax);
    sourceContainer->getTHn(step2)->GetAxis(0)->SetRangeUser(mEtaMin, mEtaMax);
  }

  Int_t dimensions[] = {0, 1, 3, 4};
  THnBase* generated = sourceContainer->getTHn(step1)->ProjectionND(4, dimensions);
  THnBase* measured = sourceContainer->getTHn(step2)->ProjectionND(4, dimensions);

  //   Printf("%d %d %f %f", step1, step2, generated->GetEntries(), measured->GetEntries());

  resetBinLimits(sourceContainer->getTHn(step1));
  resetBinLimits(sourceContainer->getTHn(step2));

  THnBase* clone = (THnBase*)measured->Clone();

  clone->Divide(measured, generated, 1, 1, "B");

  delete generated;
  delete measured;

  return clone;
}

//____________________________________________________________________
TH1* CorrelationContainer::getTrackEfficiency(CFStep step1, CFStep step2, Int_t axis1, Int_t axis2, Int_t source, Int_t axis3)
{
  // creates a track-level efficiency by dividing step2 by step1
  // projected to axis1 and axis2 (optional if >= 0)
  //
  // source: 0 = mTrackHist; 1 = mTrackHistEfficiency; 2 = mTrackHistEfficiency rebinned for pT,T / pT,lead binning

  // cache it for efficiency (usually more than one efficiency is requested)

  StepTHn* sourceContainer = nullptr;

  if (source == 0) {
    return nullptr;
  } else if (source == 1 || source == 2) {
    sourceContainer = mTrackHistEfficiency;
    // step offset because we start with kCFStepAnaTopology
    step1 = (CFStep)((Int_t)step1 - (Int_t)kCFStepAnaTopology);
    step2 = (CFStep)((Int_t)step2 - (Int_t)kCFStepAnaTopology);
  } else {
    return nullptr;
  }

  // reset all limits and set the right ones except those in axis1, axis2 and axis3
  resetBinLimits(sourceContainer->getTHn(step1));
  resetBinLimits(sourceContainer->getTHn(step2));
  if (mEtaMax > mEtaMin && axis1 != 0 && axis2 != 0 && axis3 != 0) {
    LOGF(info, "Restricted eta-range to %f %f", mEtaMin, mEtaMax);
    sourceContainer->getTHn(step1)->GetAxis(0)->SetRangeUser(mEtaMin, mEtaMax);
    sourceContainer->getTHn(step2)->GetAxis(0)->SetRangeUser(mEtaMin, mEtaMax);
  }
  if (mPtMax > mPtMin && axis1 != 1 && axis2 != 1 && axis3 != 1) {
    LOGF(info, "Restricted pt-range to %f %f", mPtMin, mPtMax);
    sourceContainer->getTHn(step1)->GetAxis(1)->SetRangeUser(mPtMin, mPtMax);
    sourceContainer->getTHn(step2)->GetAxis(1)->SetRangeUser(mPtMin, mPtMax);
  }
  if (mPartSpecies != -1 && axis1 != 2 && axis2 != 2 && axis3 != 2) {
    LOGF(info, "Restricted to particle species %d", mPartSpecies);
    sourceContainer->getTHn(step1)->GetAxis(2)->SetRangeUser(mPartSpecies, mPartSpecies);
    sourceContainer->getTHn(step2)->GetAxis(2)->SetRangeUser(mPartSpecies, mPartSpecies);
  }
  if (mCentralityMax > mCentralityMin && axis1 != 3 && axis2 != 3 && axis3 != 3) {
    LOGF(info, "Restricted centrality range to %f %f", mCentralityMin, mCentralityMax);
    sourceContainer->getTHn(step1)->GetAxis(3)->SetRangeUser(mCentralityMin, mCentralityMax);
    sourceContainer->getTHn(step2)->GetAxis(3)->SetRangeUser(mCentralityMin, mCentralityMax);
  }
  if (mZVtxMax > mZVtxMin && axis1 != 4 && axis2 != 4 && axis3 != 4) {
    LOGF(info, "Restricted z-vtx range to %f %f", mZVtxMin, mZVtxMax);
    sourceContainer->getTHn(step1)->GetAxis(4)->SetRangeUser(mZVtxMin, mZVtxMax);
    sourceContainer->getTHn(step2)->GetAxis(4)->SetRangeUser(mZVtxMin, mZVtxMax);
  }

  TH1* measured = nullptr;
  TH1* generated = nullptr;

  if (axis3 >= 0) {
    generated = sourceContainer->getTHn(step1)->Projection(axis1, axis2, axis3);
    measured = sourceContainer->getTHn(step2)->Projection(axis1, axis2, axis3);
  } else if (axis2 >= 0) {
    generated = sourceContainer->getTHn(step1)->Projection(axis1, axis2);
    measured = sourceContainer->getTHn(step2)->Projection(axis1, axis2);
  } else {
    generated = sourceContainer->getTHn(step1)->Projection(axis1);
    measured = sourceContainer->getTHn(step2)->Projection(axis1);
  }

  // check for bins with less than 50 entries, print warning
  Int_t binBegin[3];
  Int_t binEnd[3];

  binBegin[0] = 1;
  binBegin[1] = 1;
  binBegin[2] = 1;

  binEnd[0] = generated->GetNbinsX();
  binEnd[1] = generated->GetNbinsY();
  binEnd[2] = generated->GetNbinsZ();

  if (mEtaMax > mEtaMin) {
    if (axis1 == 0) {
      binBegin[0] = generated->GetXaxis()->FindBin(mEtaMin);
      binEnd[0] = generated->GetXaxis()->FindBin(mEtaMax);
    }
    if (axis2 == 0) {
      binBegin[1] = generated->GetYaxis()->FindBin(mEtaMin);
      binEnd[1] = generated->GetYaxis()->FindBin(mEtaMax);
    }
    if (axis3 == 0) {
      binBegin[2] = generated->GetZaxis()->FindBin(mEtaMin);
      binEnd[2] = generated->GetZaxis()->FindBin(mEtaMax);
    }
  }

  if (mPtMax > mPtMin) {
    // TODO this is just checking up to 15 for now
    Float_t ptMax = TMath::Min((Float_t)15., mPtMax);
    if (axis1 == 1) {
      binBegin[0] = generated->GetXaxis()->FindBin(mPtMin);
      binEnd[0] = generated->GetXaxis()->FindBin(ptMax);
    }
    if (axis2 == 1) {
      binBegin[1] = generated->GetYaxis()->FindBin(mPtMin);
      binEnd[1] = generated->GetYaxis()->FindBin(ptMax);
    }
    if (axis3 == 1) {
      binBegin[2] = generated->GetZaxis()->FindBin(mPtMin);
      binEnd[2] = generated->GetZaxis()->FindBin(ptMax);
    }
  }

  Int_t total = 0;
  Int_t count = 0;
  Int_t vars[3];

  for (Int_t i = 0; i < 3; i++) {
    vars[i] = binBegin[i];
  }

  const Int_t limit = 50;
  while (1) {
    if (generated->GetDimension() == 1 && generated->GetBinContent(vars[0]) < limit) {
      LOGF(info, "Empty bin at %s=%.2f (%.2f entries)", generated->GetXaxis()->GetTitle(), generated->GetXaxis()->GetBinCenter(vars[0]), generated->GetBinContent(vars[0]));
      count++;
    } else if (generated->GetDimension() == 2 && generated->GetBinContent(vars[0], vars[1]) < limit) {
      LOGF(info, "Empty bin at %s=%.2f %s=%.2f (%.2f entries)",
           generated->GetXaxis()->GetTitle(), generated->GetXaxis()->GetBinCenter(vars[0]),
           generated->GetYaxis()->GetTitle(), generated->GetYaxis()->GetBinCenter(vars[1]),
           generated->GetBinContent(vars[0], vars[1]));
      count++;
    } else if (generated->GetDimension() == 3 && generated->GetBinContent(vars[0], vars[1], vars[2]) < limit) {
      LOGF(info, "Empty bin at %s=%.2f %s=%.2f %s=%.2f (%.2f entries)",
           generated->GetXaxis()->GetTitle(), generated->GetXaxis()->GetBinCenter(vars[0]),
           generated->GetYaxis()->GetTitle(), generated->GetYaxis()->GetBinCenter(vars[1]),
           generated->GetZaxis()->GetTitle(), generated->GetZaxis()->GetBinCenter(vars[2]),
           generated->GetBinContent(vars[0], vars[1], vars[2]));
      count++;
    }

    vars[2]++;
    if (vars[2] == binEnd[2] + 1) {
      vars[2] = binBegin[2];
      vars[1]++;
    }

    if (vars[1] == binEnd[1] + 1) {
      vars[1] = binBegin[1];
      vars[0]++;
    }

    if (vars[0] == binEnd[0] + 1) {
      break;
    }
    total++;
  }

  LOGF(info, "Correction has %d empty bins (out of %d bins)", count, total);

  // rebin if required
  if (source == 2) {
    TAxis* axis = mTriggerHist->getTHn(0)->GetAxis(0);

    if (axis->GetNbins() < measured->GetNbinsX()) {
      if (axis2 != -1) {
        // 2d rebin with variable axis does not exist in root

        TH1* tmp = measured;
        measured = new TH2D(Form("%s_rebinned", tmp->GetName()), tmp->GetTitle(), axis->GetNbins(), axis->GetXbins()->GetArray(), tmp->GetNbinsY(), tmp->GetYaxis()->GetXbins()->GetArray());
        for (Int_t x = 1; x <= tmp->GetNbinsX(); x++) {
          for (Int_t y = 1; y <= tmp->GetNbinsY(); y++) {
            ((TH2*)measured)->Fill(tmp->GetXaxis()->GetBinCenter(x), tmp->GetYaxis()->GetBinCenter(y), tmp->GetBinContent(x, y));
            measured->SetBinError(x, y, 0); // cannot trust bin error, set to 0
          }
        }
        delete tmp;

        tmp = generated;
        generated = new TH2D(Form("%s_rebinned", tmp->GetName()), tmp->GetTitle(), axis->GetNbins(), axis->GetXbins()->GetArray(), tmp->GetNbinsY(), tmp->GetYaxis()->GetXbins()->GetArray());
        for (Int_t x = 1; x <= tmp->GetNbinsX(); x++) {
          for (Int_t y = 1; y <= tmp->GetNbinsY(); y++) {
            ((TH2*)generated)->Fill(tmp->GetXaxis()->GetBinCenter(x), tmp->GetYaxis()->GetBinCenter(y), tmp->GetBinContent(x, y));
            generated->SetBinError(x, y, 0); // cannot trust bin error, set to 0
          }
        }
        delete tmp;
      } else {
        TH1* tmp = measured;
        measured = tmp->Rebin(axis->GetNbins(), Form("%s_rebinned", tmp->GetName()), axis->GetXbins()->GetArray());
        delete tmp;

        tmp = generated;
        generated = tmp->Rebin(axis->GetNbins(), Form("%s_rebinned", tmp->GetName()), axis->GetXbins()->GetArray());
        delete tmp;
      }
    } else if (axis->GetNbins() > measured->GetNbinsX()) {
      if (axis2 != -1) {
        LOGF(fatal, "Rebinning only works for 1d at present");
      }

      // this is an unfortunate case where the number of bins has to be increased in principle
      // there is a region where the binning is finner in one histogram and a region where it is the other way round
      // this cannot be resolved in principle, but as we only calculate the ratio the bin in the second region get the same entries
      // only a certain binning is supported here
      if (axis->GetNbins() != 100 || measured->GetNbinsX() != 39) {
        LOGF(fatal, "Invalid binning --> %d %d", axis->GetNbins(), measured->GetNbinsX());
      }

      Double_t newBins[] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 100.0};

      // reduce binning below 5 GeV/c
      TH1* tmp = measured;
      measured = tmp->Rebin(27, Form("%s_rebinned", tmp->GetName()), newBins);
      delete tmp;

      // increase binning above 5 GeV/c
      tmp = measured;
      measured = new TH1F(Form("%s_rebinned2", tmp->GetName()), tmp->GetTitle(), axis->GetNbins(), axis->GetBinLowEdge(1), axis->GetBinUpEdge(axis->GetNbins()));
      for (Int_t bin = 1; bin <= measured->GetNbinsX(); bin++) {
        measured->SetBinContent(bin, tmp->GetBinContent(tmp->FindBin(measured->GetBinCenter(bin))));
        measured->SetBinError(bin, tmp->GetBinError(tmp->FindBin(measured->GetBinCenter(bin))));
      }
      delete tmp;

      // reduce binning below 5 GeV/c
      tmp = generated;
      generated = tmp->Rebin(27, Form("%s_rebinned", tmp->GetName()), newBins);
      delete tmp;

      // increase binning above 5 GeV/c
      tmp = generated;
      generated = new TH1F(Form("%s_rebinned2", tmp->GetName()), tmp->GetTitle(), axis->GetNbins(), axis->GetBinLowEdge(1), axis->GetBinUpEdge(axis->GetNbins()));
      for (Int_t bin = 1; bin <= generated->GetNbinsX(); bin++) {
        generated->SetBinContent(bin, tmp->GetBinContent(tmp->FindBin(generated->GetBinCenter(bin))));
        generated->SetBinError(bin, tmp->GetBinError(tmp->FindBin(generated->GetBinCenter(bin))));
      }
      delete tmp;
    }
  }

  measured->Divide(measured, generated, 1, 1, "B");

  delete generated;

  resetBinLimits(sourceContainer->getTHn(step1));
  resetBinLimits(sourceContainer->getTHn(step2));

  return measured;
}

//____________________________________________________________________
TH1* CorrelationContainer::getEventEfficiency(CFStep step1, CFStep step2, Int_t axis1, Int_t axis2, Float_t ptTriggerMin, Float_t ptTriggerMax)
{
  // creates a event-level efficiency by dividing step2 by step1
  // projected to axis1 and axis2 (optional if >= 0)

  if (ptTriggerMax > ptTriggerMin) {
    mTriggerHist->getTHn(step1)->GetAxis(0)->SetRangeUser(ptTriggerMin, ptTriggerMax);
    mTriggerHist->getTHn(step2)->GetAxis(0)->SetRangeUser(ptTriggerMin, ptTriggerMax);
  }

  TH1* measured = nullptr;
  TH1* generated = nullptr;

  if (axis2 >= 0) {
    generated = mTriggerHist->getTHn(step1)->Projection(axis1, axis2);
    measured = mTriggerHist->getTHn(step2)->Projection(axis1, axis2);
  } else {
    generated = mTriggerHist->getTHn(step1)->Projection(axis1);
    measured = mTriggerHist->getTHn(step2)->Projection(axis1);
  }

  measured->Divide(measured, generated, 1, 1, "B");

  delete generated;

  if (ptTriggerMax > ptTriggerMin) {
    mTriggerHist->getTHn(step1)->GetAxis(0)->SetRangeUser(0, -1);
    mTriggerHist->getTHn(step2)->GetAxis(0)->SetRangeUser(0, -1);
  }

  return measured;
}

//____________________________________________________________________
void CorrelationContainer::weightHistogram(TH3* hist1, TH1* hist2)
{
  // weights each entry of the 3d histogram hist1 with the 1d histogram hist2
  // where the matching is done of the z axis of hist1 with the x axis of hist2

  if (hist1->GetNbinsZ() != hist2->GetNbinsX()) {
    LOGF(fatal, "Inconsistent binning %d %d", hist1->GetNbinsZ(), hist2->GetNbinsX());
  }

  for (Int_t x = 1; x <= hist1->GetNbinsX(); x++) {
    for (Int_t y = 1; y <= hist1->GetNbinsY(); y++) {
      for (Int_t z = 1; z <= hist1->GetNbinsZ(); z++) {
        if (hist2->GetBinContent(z) > 0) {
          hist1->SetBinContent(x, y, z, hist1->GetBinContent(x, y, z) / hist2->GetBinContent(z));
          hist1->SetBinError(x, y, z, hist1->GetBinError(x, y, z) / hist2->GetBinContent(z));
        } else {
          hist1->SetBinContent(x, y, z, 0);
          hist1->SetBinError(x, y, z, 0);
        }
      }
    }
  }
}

//____________________________________________________________________
TH1* CorrelationContainer::getBias(CFStep step1, CFStep step2, const char* axis, Float_t leadPtMin, Float_t leadPtMax, Int_t weighting)
{
  // extracts the track-level bias (integrating out the multiplicity) between two steps (dividing step2 by step1)
  // done by weighting the track-level distribution with the number of events as function of leading pT
  // and then calculating the ratio between the distributions
  // projected to axis which is a TH3::Project3D string, e.g. "x", or "yx"
  //   no projection is done if axis == 0
  // weighting: 0 = tracks weighted with events (as discussed above)
  //            1 = only track bias is returned
  //            2 = only event bias is returned

  StepTHn* tmp = mPairHist;

  resetBinLimits(tmp->getTHn(step1));
  resetBinLimits(mTriggerHist->getTHn(step1));
  setBinLimits(tmp->getTHn(step1));

  resetBinLimits(tmp->getTHn(step2));
  resetBinLimits(mTriggerHist->getTHn(step2));
  setBinLimits(tmp->getTHn(step2));

  TH1D* events1 = (TH1D*)mTriggerHist->getTHn(step1)->Projection(0);
  TH3D* hist1 = (TH3D*)tmp->getTHn(step1)->Projection(0, tmp->getNVar() - 1, 2);
  if (weighting == 0) {
    weightHistogram(hist1, events1);
  }

  TH1D* events2 = (TH1D*)mTriggerHist->getTHn(step2)->Projection(0);
  TH3D* hist2 = (TH3D*)tmp->getTHn(step2)->Projection(0, tmp->getNVar() - 1, 2);
  if (weighting == 0) {
    weightHistogram(hist2, events2);
  }

  TH1* generated = hist1;
  TH1* measured = hist2;

  if (weighting == 0 || weighting == 1) {
    if (axis) {
      if (leadPtMax > leadPtMin) {
        hist1->GetZaxis()->SetRangeUser(leadPtMin, leadPtMax);
        hist2->GetZaxis()->SetRangeUser(leadPtMin, leadPtMax);
      }

      if (mEtaMax > mEtaMin && !TString(axis).Contains("x")) {
        hist1->GetXaxis()->SetRangeUser(mEtaMin, mEtaMax);
        hist2->GetXaxis()->SetRangeUser(mEtaMin, mEtaMax);
      }

      generated = hist1->Project3D(axis);
      measured = hist2->Project3D(axis);

      // delete hists here if projection has been done
      delete hist1;
      delete hist2;
    }
    delete events1;
    delete events2;
  } else if (weighting == 2) {
    delete hist1;
    delete hist2;
    generated = events1;
    measured = events2;
  }

  measured->Divide(generated);

  delete generated;

  resetBinLimits(tmp->getTHn(step1));
  resetBinLimits(tmp->getTHn(step2));

  return measured;
}

//____________________________________________________________________
TH2D* CorrelationContainer::getTrackingEfficiency()
{
  // extracts the tracking efficiency by calculating the efficiency from step kCFStepAnaTopology to kCFStepTrackedOnlyPrim
  // integrates over the regions and all other variables than pT and eta to increase the statistics
  //
  // returned histogram has to be deleted by the user

  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepAnaTopology, kCFStepTrackedOnlyPrim, 0, 1));
}

//____________________________________________________________________
TH2D* CorrelationContainer::getFakeRate()
{
  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepTracked, (CFStep)(kCFStepTracked + 3), 0, 1));
}

//____________________________________________________________________
TH2D* CorrelationContainer::getTrackingEfficiencyCentrality()
{
  // extracts the tracking efficiency by calculating the efficiency from step kCFStepAnaTopology to kCFStepTrackedOnlyPrim
  // integrates over the regions and all other variables than pT, centrality to increase the statistics
  //
  // returned histogram has to be deleted by the user

  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepAnaTopology, kCFStepTrackedOnlyPrim, 1, 3));
}

//____________________________________________________________________
TH1D* CorrelationContainer::getTrackingEfficiency(Int_t axis)
{
  // extracts the tracking efficiency by calculating the efficiency from step kCFStepAnaTopology to kCFStepTrackedOnlyPrim
  // integrates over the regions and all other variables than pT (axis == 0) and eta (axis == 1) to increase the statistics

  return dynamic_cast<TH1D*>(getTrackEfficiency(kCFStepAnaTopology, kCFStepTrackedOnlyPrim, axis));
}

//____________________________________________________________________
TH1D* CorrelationContainer::getFakeRate(Int_t axis)
{
  return dynamic_cast<TH1D*>(getTrackEfficiency(kCFStepTracked, (CFStep)(kCFStepTracked + 3), axis));
}
//____________________________________________________________________
TH2D* CorrelationContainer::getTrackingCorrection()
{
  // extracts the tracking correction by calculating the efficiency from step kCFStepAnaTopology to kCFStepTracked
  // integrates over the regions and all other variables than pT and eta to increase the statistics
  //
  // returned histogram has to be deleted by the user

  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepTracked, kCFStepAnaTopology, 0, 1));
}

//____________________________________________________________________
TH1D* CorrelationContainer::getTrackingCorrection(Int_t axis)
{
  // extracts the tracking correction by calculating the efficiency from step kCFStepAnaTopology to kCFStepTracked
  // integrates over the regions and all other variables than pT (axis == 0) and eta (axis == 1) to increase the statistics

  return dynamic_cast<TH1D*>(getTrackEfficiency(kCFStepTracked, kCFStepAnaTopology, axis));
}

//____________________________________________________________________
TH2D* CorrelationContainer::getTrackingEfficiencyCorrection()
{
  // extracts the tracking correction by calculating the efficiency from step kCFStepAnaTopology to kCFStepTracked
  // integrates over the regions and all other variables than pT and eta to increase the statistics
  //
  // returned histogram has to be deleted by the user

  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepTrackedOnlyPrim, kCFStepAnaTopology, 0, 1));
}

//____________________________________________________________________
TH2D* CorrelationContainer::getTrackingEfficiencyCorrectionCentrality()
{
  // extracts the tracking correction by calculating the efficiency from step kCFStepAnaTopology to kCFStepTracked
  // integrates over the regions and all other variables than pT and centrality to increase the statistics
  //
  // returned histogram has to be deleted by the user

  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepTrackedOnlyPrim, kCFStepAnaTopology, 1, 3));
}

//____________________________________________________________________
TH1D* CorrelationContainer::getTrackingEfficiencyCorrection(Int_t axis)
{
  // extracts the tracking correction by calculating the efficiency from step kCFStepAnaTopology to kCFStepTracked
  // integrates over the regions and all other variables than pT (axis == 0) and eta (axis == 1) to increase the statistics

  return dynamic_cast<TH1D*>(getTrackEfficiency(kCFStepTrackedOnlyPrim, kCFStepAnaTopology, axis));
}

//____________________________________________________________________
TH2D* CorrelationContainer::getTrackingContamination()
{
  // extracts the tracking contamination by secondaries by calculating the efficiency from step kCFStepTrackedOnlyPrim to kCFStepTracked
  // integrates over the regions and all other variables than pT and eta to increase the statistics
  //
  // returned histogram has to be deleted by the user

  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepTracked, kCFStepTrackedOnlyPrim, 0, 1));
}

//____________________________________________________________________
TH2D* CorrelationContainer::getTrackingContaminationCentrality()
{
  // extracts the tracking contamination by secondaries by calculating the efficiency from step kCFStepTrackedOnlyPrim to kCFStepTracked
  // integrates over the regions and all other variables than pT and centrality to increase the statistics
  //
  // returned histogram has to be deleted by the user

  return dynamic_cast<TH2D*>(getTrackEfficiency(kCFStepTracked, kCFStepTrackedOnlyPrim, 1, 3));
}

//____________________________________________________________________
TH1D* CorrelationContainer::getTrackingContamination(Int_t axis)
{
  // extracts the tracking contamination by secondaries by calculating the efficiency from step kCFStepTrackedOnlyPrim to kCFStepTracked
  // integrates over the regions and all other variables than pT (axis == 0) and eta (axis == 1) to increase the statistics

  return dynamic_cast<TH1D*>(getTrackEfficiency(kCFStepTracked, kCFStepTrackedOnlyPrim, axis));
}

//____________________________________________________________________
const char* CorrelationContainer::getStepTitle(CFStep step)
{
  // returns the name of the given step

  switch (step) {
    case kCFStepAll:
      return "All events";
    case kCFStepTriggered:
      return "Triggered";
    case kCFStepVertex:
      return "Primary Vertex";
    case kCFStepAnaTopology:
      return "Required analysis topology";
    case kCFStepTrackedOnlyPrim:
      return "Tracked (matched MC, only primaries)";
    case kCFStepTracked:
      return "Tracked (matched MC, all)";
    case kCFStepReconstructed:
      return "Reconstructed";
    case kCFStepRealLeading:
      return "Correct leading particle identified";
    case kCFStepBiasStudy:
      return "Bias study applying tracking efficiency";
    case kCFStepBiasStudy2:
      return "Bias study applying tracking efficiency in two steps";
    case kCFStepCorrected:
      return "Corrected for efficiency on-the-fly";
  }

  return "";
}

void CorrelationContainer::deepCopy(CorrelationContainer* from)
{
  // copies the entries of this object's members from the object <from> to this object
  // fills using the fill function and thus allows that the objects have different binning

  for (Int_t step = 0; step < mPairHist->getNSteps(); step++) {
    LOGF(info, "Copying step %d", step);
    THnBase* target = mPairHist->getTHn(step);
    THnBase* source = from->mPairHist->getTHn(step);

    target->Reset();
    target->RebinnedAdd(source);
  }

  for (Int_t step = 0; step < mTriggerHist->getNSteps(); step++) {
    LOGF(info, "Ev: Copying step %d", step);
    THnBase* target = mTriggerHist->getTHn(step);
    THnBase* source = from->mTriggerHist->getTHn(step);

    target->Reset();
    target->RebinnedAdd(source);
  }

  for (Int_t step = 0; step < TMath::Min(mTrackHistEfficiency->getNSteps(), from->mTrackHistEfficiency->getNSteps()); step++) {
    if (!from->mTrackHistEfficiency->getTHn(step)) {
      continue;
    }

    LOGF(info, "Eff: Copying step %d", step);
    THnBase* target = mTrackHistEfficiency->getTHn(step);
    THnBase* source = from->mTrackHistEfficiency->getTHn(step);

    target->Reset();
    target->RebinnedAdd(source);
  }
}

void CorrelationContainer::symmetrizepTBins()
{
  // copy pt,a < pt,t bins to pt,a > pt,t (inverting deltaphi and delta eta as it should be) including symmetric bins

  for (Int_t step = 0; step < mPairHist->getNSteps(); step++) {
    LOGF(info, "Copying step %d", step);
    THnBase* target = mPairHist->getTHn(step);
    if (target->GetEntries() == 0) {
      continue;
    }

    // for symmetric bins
    THnBase* source = (THnBase*)target->Clone();

    Int_t zVtxBins = 1;
    if (target->GetNdimensions() > 5) {
      zVtxBins = target->GetAxis(5)->GetNbins();
    }

    // axes: 0 delta eta; 1 pT,a; 2 pT,t; 3 centrality; 4 delta phi; 5 vtx-z
    for (Int_t i3 = 1; i3 <= target->GetAxis(3)->GetNbins(); i3++) {
      for (Int_t i5 = 1; i5 <= zVtxBins; i5++) {
        for (Int_t i1 = 1; i1 <= target->GetAxis(1)->GetNbins(); i1++) {
          for (Int_t i2 = 1; i2 <= target->GetAxis(2)->GetNbins(); i2++) {
            // find source bin
            Int_t binA = target->GetAxis(1)->FindBin(target->GetAxis(2)->GetBinCenter(i2));
            Int_t binT = target->GetAxis(2)->FindBin(target->GetAxis(1)->GetBinCenter(i1));

            LOGF(info, "(%d %d) Copying from %d %d to %d %d", i3, i5, binA, binT, i1, i2);

            for (Int_t i0 = 1; i0 <= target->GetAxis(0)->GetNbins(); i0++) {
              for (Int_t i4 = 1; i4 <= target->GetAxis(4)->GetNbins(); i4++) {
                Int_t binEta = target->GetAxis(0)->FindBin(-target->GetAxis(0)->GetBinCenter(i0));
                Double_t phi = -target->GetAxis(4)->GetBinCenter(i4);
                if (phi < -M_PI / 2) {
                  phi += M_PI * 2;
                }
                Int_t binPhi = target->GetAxis(4)->FindBin(phi);

                Int_t binSource[] = {binEta, binA, binT, i3, binPhi, i5};
                Int_t binTarget[] = {i0, i1, i2, i3, i4, i5};

                Double_t value = source->GetBinContent(binSource);
                Double_t error = source->GetBinError(binSource);

                if (error == 0) {
                  continue;
                }

                Double_t value2 = target->GetBinContent(binTarget);
                Double_t error2 = target->GetBinError(binTarget);

                Double_t sum = value;
                Double_t err = error;

                if (error2 > 0) {
                  sum = value + value2;
                  err = TMath::Sqrt(error * error + error2 * error2);
                }

                // Printf("  Values: %f +- %f; %f +- %f --> %f +- %f", value, error, value2, error2, sum, err);

                target->SetBinContent(binTarget, sum);
                target->SetBinError(binTarget, err);
              }
            }
          }
        }
      }
    }

    delete source;
  }
}

//____________________________________________________________________
void CorrelationContainer::extendTrackingEfficiency(Bool_t verbose)
{
  // fits the tracking efficiency at high pT with a constant and fills all bins with this tracking efficiency

  Float_t fitRangeBegin = 5.01;
  Float_t fitRangeEnd = 14.99;
  Float_t extendRangeBegin = 10.01;

  if (mTrackHistEfficiency->getNVar() == 3) {
    TH1* obj = getTrackingEfficiency(1);

    if (verbose) {
      new TCanvas;
      obj->Draw();
    }

    obj->Fit("pol0", (verbose) ? "+" : "0+", "SAME", fitRangeBegin, fitRangeEnd);

    Float_t trackingEff = obj->GetFunction("pol0")->GetParameter(0);

    obj = getTrackingContamination(1);

    if (verbose) {
      new TCanvas;
      obj->Draw();
    }

    obj->Fit("pol0", (verbose) ? "+" : "0+", "SAME", fitRangeBegin, fitRangeEnd);

    Float_t trackingCont = obj->GetFunction("pol0")->GetParameter(0);

    LOGF(info, "CorrelationContainer::extendTrackingEfficiency: Fitted efficiency between %f and %f and got %f tracking efficiency and %f tracking contamination correction. Extending from %f onwards (within %f < eta < %f)", fitRangeBegin, fitRangeEnd, trackingEff, trackingCont, extendRangeBegin, mEtaMin, mEtaMax);

    // extend for full pT range
    for (Int_t x = mTrackHistEfficiency->getTHn(0)->GetAxis(0)->FindBin(mEtaMin); x <= mTrackHistEfficiency->getTHn(0)->GetAxis(0)->FindBin(mEtaMax); x++) {
      for (Int_t y = mTrackHistEfficiency->getTHn(0)->GetAxis(1)->FindBin(extendRangeBegin); y <= mTrackHistEfficiency->getTHn(0)->GetAxis(1)->GetNbins(); y++) {
        for (Int_t z = 1; z <= mTrackHistEfficiency->getTHn(0)->GetAxis(2)->GetNbins(); z++) // particle type axis
        {

          Int_t bins[3];
          bins[0] = x;
          bins[1] = y;
          bins[2] = z;

          mTrackHistEfficiency->getTHn(0)->SetBinContent(bins, 100);
          mTrackHistEfficiency->getTHn(1)->SetBinContent(bins, 100.0 * trackingEff);
          mTrackHistEfficiency->getTHn(2)->SetBinContent(bins, 100.0 * trackingEff / trackingCont);
        }
      }
    }
  } else if (mTrackHistEfficiency->getNVar() == 4) {
    // fit in centrality intervals of 20% for efficiency, one bin for contamination
    Float_t* trackingEff = nullptr;
    Float_t* trackingCont = nullptr;
    Float_t centralityBins[] = {0, 10, 20, 40, 60, 100};
    Int_t nCentralityBins = 5;

    LOGF(info, "CorrelationContainer::extendTrackingEfficiency: Fitting efficiencies between %f and %f. Extending from %f onwards (within %f < eta < %f)", fitRangeBegin, fitRangeEnd, extendRangeBegin, mEtaMin, mEtaMax);

    // 0 = eff; 1 = cont
    for (Int_t caseNo = 0; caseNo < 2; caseNo++) {
      Float_t* target = nullptr;
      Int_t centralityBinsLocal = nCentralityBins;

      if (caseNo == 0) {
        trackingEff = new Float_t[centralityBinsLocal];
        target = trackingEff;
      } else {
        centralityBinsLocal = 1;
        trackingCont = new Float_t[centralityBinsLocal];
        target = trackingCont;
      }

      for (Int_t i = 0; i < centralityBinsLocal; i++) {
        if (centralityBinsLocal == 1) {
          setCentralityRange(centralityBins[0] + 0.1, centralityBins[nCentralityBins] - 0.1);
        } else {
          setCentralityRange(centralityBins[i] + 0.1, centralityBins[i + 1] - 0.1);
        }
        TH1* proj = (caseNo == 0) ? getTrackingEfficiency(1) : getTrackingContamination(1);
        if (verbose) {
          new TCanvas;
          proj->DrawCopy();
        }
        if ((Int_t)proj->Fit("pol0", (verbose) ? "+" : "Q0+", "SAME", fitRangeBegin, fitRangeEnd) == 0) {
          target[i] = proj->GetFunction("pol0")->GetParameter(0);
        } else {
          target[i] = 0;
        }
        LOGF(info, "CorrelationContainer::extendTrackingEfficiency: case %d, centrality %d, eff %f", caseNo, i, target[i]);
      }
    }

    // extend for full pT range
    for (Int_t x = mTrackHistEfficiency->getTHn(0)->GetAxis(0)->FindBin(mEtaMin); x <= mTrackHistEfficiency->getTHn(0)->GetAxis(0)->FindBin(mEtaMax); x++) {
      for (Int_t y = mTrackHistEfficiency->getTHn(0)->GetAxis(1)->FindBin(extendRangeBegin); y <= mTrackHistEfficiency->getTHn(0)->GetAxis(1)->GetNbins(); y++) {
        for (Int_t z = 1; z <= mTrackHistEfficiency->getTHn(0)->GetAxis(2)->GetNbins(); z++) // particle type axis
        {
          for (Int_t z2 = 1; z2 <= mTrackHistEfficiency->getTHn(0)->GetAxis(3)->GetNbins(); z2++) // centrality axis
          {

            Int_t bins[4];
            bins[0] = x;
            bins[1] = y;
            bins[2] = z;
            bins[3] = z2;

            Int_t z2Bin = 0;
            while (centralityBins[z2Bin + 1] < mTrackHistEfficiency->getTHn(0)->GetAxis(3)->GetBinCenter(z2)) {
              z2Bin++;
            }

            //Printf("%d %d", z2, z2Bin);

            mTrackHistEfficiency->getTHn(0)->SetBinContent(bins, 100);
            mTrackHistEfficiency->getTHn(1)->SetBinContent(bins, 100.0 * trackingEff[z2Bin]);
            if (trackingCont[0] > 0) {
              mTrackHistEfficiency->getTHn(2)->SetBinContent(bins, 100.0 * trackingEff[z2Bin] / trackingCont[0]);
            } else {
              mTrackHistEfficiency->getTHn(2)->SetBinContent(bins, 0);
            }
          }
        }
      }
    }

    delete[] trackingEff;
    delete[] trackingCont;
  }

  setCentralityRange(0, 0);
}

/*
void CorrelationContainer::Scale(Double_t factor)
{
  // scales all contained histograms by the given factor

  for (Int_t i=0; i<4; i++)
    if (mTrackHist[i])
      mTrackHist[i]->Scale(factor);

  mEventHist->Scale(factor);
  mTrackHistEfficiency->Scale(factor);
}*/

void CorrelationContainer::Reset()
{
  // resets all contained histograms

  for (Int_t step = 0; step < mPairHist->getNSteps(); step++) {
    mPairHist->getTHn(step)->Reset();
  }

  for (Int_t step = 0; step < mTriggerHist->getNSteps(); step++) {
    mTriggerHist->getTHn(step)->Reset();
  }

  for (Int_t step = 0; step < mTrackHistEfficiency->getNSteps(); step++) {
    mTrackHistEfficiency->getTHn(step)->Reset();
  }
}

THnBase* CorrelationContainer::changeToThn(THnBase* sparse)
{
  // change the object to THn for faster processing

  return THn::CreateHn(Form("%s_thn", sparse->GetName()), sparse->GetTitle(), sparse);
}

void CorrelationContainer::fillEvent(Float_t centrality, CFStep step)
{
  // Fill per-event information
  mEventCount->Fill(step, centrality);
}
