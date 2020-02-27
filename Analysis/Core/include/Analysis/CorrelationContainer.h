// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef CorrelationContainer_H
#define CorrelationContainer_H

// encapsulate histogram and corrections for correlation analysis

#include "TNamed.h"
#include "TString.h"

class TH1;
class TH1F;
class TH3;
class TH3F;
class TH1D;
class TH2;
class TH2D;
class TCollection;
class THnSparse;
class THnBase;
class StepTHnBase;

class CorrelationContainer : public TNamed
{
 public:
  CorrelationContainer();
  CorrelationContainer(const char* name, const char* objTitle, const char* reqHist = "", const char* binning = nullptr);
  virtual ~CorrelationContainer(); // NOLINT: Making this override breaks compilation for unknown reason

  static const Int_t fgkCFSteps;
  enum CFStep { kCFStepAll = 0,
                kCFStepTriggered,
                kCFStepVertex,
                kCFStepAnaTopology,
                kCFStepTrackedOnlyPrim,
                kCFStepTracked,
                kCFStepReconstructed,
                kCFStepRealLeading,
                kCFStepBiasStudy,
                kCFStepBiasStudy2,
                kCFStepCorrected };

  const char* getStepTitle(CFStep step);

  StepTHnBase* getTrackHist() { return mTrackHist; }
  StepTHnBase* getEventHist() { return mEventHist; }
  StepTHnBase* getTrackHistEfficiency() { return mTrackHistEfficiency; }

  void setTrackHist(StepTHnBase* hist) { mTrackHist = hist; }
  void setEventHist(StepTHnBase* hist) { mEventHist = hist; }
  void setTrackHistEfficiency(StepTHnBase* hist) { mTrackHistEfficiency = hist; }

  void deepCopy(CorrelationContainer* from);

  void getHistsZVtxMult(CorrelationContainer::CFStep step, Float_t ptLeadMin, Float_t ptLeadMax, THnBase** trackHist, TH2** eventHist);
  TH2* getSumOfRatios(CorrelationContainer* mixed, CorrelationContainer::CFStep step, Float_t ptLeadMin, Float_t ptLeadMax, Int_t multBinBegin, Int_t multBinEnd, Bool_t normalizePerTrigger = kTRUE, Int_t stepForMixed = -1, Int_t* trigger = nullptr);
  TH1* getTriggersAsFunctionOfMultiplicity(CorrelationContainer::CFStep step, Float_t ptLeadMin, Float_t ptLeadMax);

  TH1* getTrackEfficiency(CFStep step1, CFStep step2, Int_t axis1, Int_t axis2 = -1, Int_t source = 1, Int_t axis3 = -1);
  THnBase* getTrackEfficiencyND(CFStep step1, CFStep step2);
  TH1* getEventEfficiency(CFStep step1, CFStep step2, Int_t axis1, Int_t axis2 = -1, Float_t ptLeadMin = -1, Float_t ptLeadMax = -1);
  TH1* getBias(CFStep step1, CFStep step2, const char* axis, Float_t leadPtMin = 0, Float_t leadPtMax = -1, Int_t weighting = 0);

  TH1D* getTrackingEfficiency(Int_t axis);
  TH2D* getTrackingEfficiency();
  TH2D* getTrackingEfficiencyCentrality();

  TH2D* getFakeRate();
  TH1D* getFakeRate(Int_t axis);

  TH1D* getTrackingContamination(Int_t axis);
  TH2D* getTrackingContamination();
  TH2D* getTrackingContaminationCentrality();

  TH1D* getTrackingCorrection(Int_t axis);
  TH2D* getTrackingCorrection();

  TH1D* getTrackingEfficiencyCorrection(Int_t axis);
  TH2D* getTrackingEfficiencyCorrection();
  TH2D* getTrackingEfficiencyCorrectionCentrality();

  void extendTrackingEfficiency(Bool_t verbose = kFALSE);

  void setEtaRange(Float_t etaMin, Float_t etaMax)
  {
    mEtaMin = etaMin;
    mEtaMax = etaMax;
  }
  void setPtRange(Float_t ptMin, Float_t ptMax)
  {
    mPtMin = ptMin;
    mPtMax = ptMax;
  }
  void setPartSpecies(Int_t species) { mPartSpecies = species; }
  void setCentralityRange(Float_t min, Float_t max)
  {
    mCentralityMin = min;
    mCentralityMax = max;
  }
  void setZVtxRange(Float_t min, Float_t max)
  {
    mZVtxMin = min;
    mZVtxMax = max;
  }
  void setPt2Min(Float_t ptMin) { mPt2Min = ptMin; }
  void setPt2Max(Float_t ptMin) { mPt2Max = ptMin; }

  Float_t getTrackEtaCut() { return mTrackEtaCut; }
  void setTrackEtaCut(Float_t value) { mTrackEtaCut = value; }
  void setWeightPerEvent(Bool_t flag) { mWeightPerEvent = flag; }
  void setSkipScaleMixedEvent(Bool_t flag) { mSkipScaleMixedEvent = flag; }

  void setHistogramType(const char* histogramType) { mHistogramType = histogramType; }

  void countEmptyBins(CorrelationContainer::CFStep step, Float_t ptLeadMin, Float_t ptLeadMax);
  void symmetrizepTBins();

  void setBinLimits(THnBase* grid);
  void resetBinLimits(THnBase* grid);

  void setGetMultCache(Bool_t flag = kTRUE) { mGetMultCacheOn = flag; }

  CorrelationContainer(const CorrelationContainer& c);
  CorrelationContainer& operator=(const CorrelationContainer& corr);
  virtual void Copy(TObject& c) const; // NOLINT: Making this override breaks compilation for unknown reason

  virtual Long64_t Merge(TCollection* list);
  //void Scale(Double_t factor);
  void Reset();
  THnBase* changeToThn(THnBase* sparse);

  static TString combineBinning(TString defaultBinning, TString customBinning);

 protected:
  Double_t* getBinning(const char* configuration, const char* tag, Int_t& nBins);
  void weightHistogram(TH3* hist1, TH1* hist2);
  void multiplyHistograms(THnBase* grid, THnBase* target, TH1* histogram, Int_t var1, Int_t var2);

  StepTHnBase* mTrackHist;           // container for track level distributions at all analysis steps
  StepTHnBase* mEventHist;           // container for event level distribution at all analysis steps
  StepTHnBase* mTrackHistEfficiency; // container for tracking efficiency and contamination (all particles filled including leading one): axes: eta, pT, particle species

  Float_t mEtaMin;        // eta min for projections
  Float_t mEtaMax;        // eta max for projections
  Float_t mPtMin;         // pT min for projections (for track pT, not pT,lead)
  Float_t mPtMax;         // pT max for projections (for track pT, not pT,lead)
  Int_t mPartSpecies;     // Particle species for projections
  Float_t mCentralityMin; // centrality min for projections
  Float_t mCentralityMax; // centrality max for projections
  Float_t mZVtxMin;       // z vtx min for projections
  Float_t mZVtxMax;       // z vtx max for projections
  Float_t mPt2Min;        // pT min for projections (for pT,2 (only 2+1 corr case))
  Float_t mPt2Max;        // pT max for projections (for pT,2 (only 2+1 corr case))

  Float_t mTrackEtaCut;        // cut used during production of histograms (needed for finite bin correction in getSumOfRatios)
  Bool_t mWeightPerEvent;      // weight with the number of trigger particles per event
  Bool_t mSkipScaleMixedEvent; // scale the mixed event with (0, 0) plus finite bin correction (default: kTRUE)

  StepTHnBase* mCache; //! cache variable for getTrackEfficiency

  Bool_t mGetMultCacheOn; //! cache for getHistsZVtxMult function active
  THnBase* mGetMultCache; //! cache for getHistsZVtxMult function

  TString mHistogramType; // what is stored in this histogram

  ClassDef(CorrelationContainer, 1) // underlying event histogram container
};

#endif
