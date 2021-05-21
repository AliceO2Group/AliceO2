// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_EVENTSELECTION_H_
#define O2_ANALYSIS_EVENTSELECTION_H_

#include "Framework/AnalysisDataModel.h"
#include "AnalysisCore/TriggerAliases.h"

namespace o2::aod
{

// Bits in eventCuts bitmask in Run2BCInfos table
// Must be consistent with EventSelectionCut enum in the Run2 converter
enum Run2EventCuts {
  kINELgtZERO = 0,
  kPileupInMultBins,
  kConsistencySPDandTrackVertices,
  kTrackletsVsClusters,
  kNonZeroNContribs,
  kIncompleteDAQ,
  kPileUpMV,
  kTPCPileUp,
  kTimeRangeCut,
  kEMCALEDCut,
  kAliEventCutsAccepted,
  kIsPileupFromSPD,
  kIsV0PFPileup,
  kIsTPCHVdip,
  kIsTPCLaserWarmUp,
  kTRDHCO, // Offline TRD cosmic trigger decision
  kTRDHJT, // Offline TRD jet trigger decision
  kTRDHSE, // Offline TRD single electron trigger decision
  kTRDHQU, // Offline TRD quarkonium trigger decision
  kTRDHEE  // Offline TRD single-electron-in-EMCAL-acceptance trigger decision
};

// Event selection criteria
enum EventSelectionFlags {
  kIsBBV0A = 0,
  kIsBBV0C,
  kIsBBFDA,
  kIsBBFDC,
  kNoBGV0A,
  kNoBGV0C,
  kNoBGFDA,
  kNoBGFDC,
  kIsBBT0A,
  kIsBBT0C,
  kIsBBZNA,
  kIsBBZNC,
  kNoBGZNA,
  kNoBGZNC,
  kNoV0MOnVsOfPileup,
  kNoSPDOnVsOfPileup,
  kNoV0Casymmetry,
  kIsGoodTimeRange,
  kNoIncompleteDAQ,
  kNoTPCLaserWarmUp,
  kNoTPCHVdip,
  kNoPileupFromSPD,
  kNoV0PFPileup,
  kNoSPDClsVsTklBG,
  kNoV0C012vsTklBG,
  kNsel
};

// collision-joinable event selection decisions
namespace evsel
{
// TODO bool arrays are not supported? Storing in int32 for the moment
DECLARE_SOA_COLUMN(Alias, alias, int32_t[kNaliases]);
DECLARE_SOA_COLUMN(Selection, selection, int32_t[kNsel]);
DECLARE_SOA_COLUMN(BBV0A, bbV0A, bool);                 //! Beam-beam time in V0A
DECLARE_SOA_COLUMN(BBV0C, bbV0C, bool);                 //! Beam-beam time in V0C
DECLARE_SOA_COLUMN(BGV0A, bgV0A, bool);                 //! Beam-gas time in V0A
DECLARE_SOA_COLUMN(BGV0C, bgV0C, bool);                 //! Beam-gas time in V0C
DECLARE_SOA_COLUMN(BBFDA, bbFDA, bool);                 //! Beam-beam time in FDA
DECLARE_SOA_COLUMN(BBFDC, bbFDC, bool);                 //! Beam-beam time in FDC
DECLARE_SOA_COLUMN(BGFDA, bgFDA, bool);                 //! Beam-gas time in FDA
DECLARE_SOA_COLUMN(BGFDC, bgFDC, bool);                 //! Beam-gas time in FDC
DECLARE_SOA_COLUMN(MultRingV0A, multRingV0A, float[5]); //! V0A multiplicity per ring (4 rings in run2, 5 rings in run3)
DECLARE_SOA_COLUMN(MultRingV0C, multRingV0C, float[4]); //! V0C multiplicity per ring (4 rings in run2)
DECLARE_SOA_COLUMN(SpdClusters, spdClusters, uint32_t); //! Number of SPD clusters in two layers
DECLARE_SOA_COLUMN(NTracklets, nTracklets, int);        //! Tracklet multiplicity
DECLARE_SOA_COLUMN(Sel7, sel7, bool);                   //! Event selection decision based on V0A & V0C
DECLARE_SOA_COLUMN(Sel8, sel8, bool);                   //! Event selection decision based on TVX
DECLARE_SOA_COLUMN(FoundFT0, foundFT0, int64_t);        //! FT0 entry index in FT0s table (-1 if doesn't exist)
} // namespace evsel
DECLARE_SOA_TABLE(EvSels, "AOD", "EVSEL", //!
                  evsel::Alias, evsel::Selection,
                  evsel::BBV0A, evsel::BBV0C, evsel::BGV0A, evsel::BGV0C,
                  evsel::BBFDA, evsel::BBFDC, evsel::BGFDA, evsel::BGFDC,
                  evsel::MultRingV0A, evsel::MultRingV0C, evsel::SpdClusters, evsel::NTracklets,
                  evsel::Sel7, evsel::Sel8, evsel::FoundFT0);
using EvSel = EvSels::iterator;

DECLARE_SOA_TABLE(BcSels, "AOD", "BCSEL", //!
                  evsel::Alias, evsel::Selection,
                  evsel::BBV0A, evsel::BBV0C, evsel::BGV0A, evsel::BGV0C,
                  evsel::BBFDA, evsel::BBFDC, evsel::BGFDA, evsel::BGFDC,
                  evsel::MultRingV0A, evsel::MultRingV0C, evsel::SpdClusters, evsel::FoundFT0);
using BcSel = BcSels::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_EVENTSELECTION_H_
