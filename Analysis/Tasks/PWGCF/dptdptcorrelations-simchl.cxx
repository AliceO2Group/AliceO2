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

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisConfigurableCuts.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include <TROOT.h>
#include <TParameter.h>
#include <TList.h>
#include <TDirectory.h>
#include <TFolder.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TProfile3D.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::soa;
using namespace o2::framework::expressions;

#include "Framework/runDataProcessing.h"

namespace o2
{
namespace aod
{
/* we have to change from int to bool when bool columns work properly */
namespace dptdptcorrelations
{
DECLARE_SOA_COLUMN(EventAccepted, eventaccepted, uint8_t);
DECLARE_SOA_COLUMN(EventCentMult, centmult, float);
DECLARE_SOA_COLUMN(TrackacceptedAsOne, trackacceptedasone, uint8_t);
DECLARE_SOA_COLUMN(TrackacceptedAsTwo, trackacceptedastwo, uint8_t);
} // namespace dptdptcorrelations
DECLARE_SOA_TABLE(AcceptedEvents, "AOD", "ACCEPTEDEVENTS", dptdptcorrelations::EventAccepted, dptdptcorrelations::EventCentMult);
DECLARE_SOA_TABLE(ScannedTracks, "AOD", "SCANNEDTRACKS", dptdptcorrelations::TrackacceptedAsOne, dptdptcorrelations::TrackacceptedAsTwo);

using CollisionEvSelCent = soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator;
using TrackData = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended, aod::TrackSelection>::iterator;
using FilteredTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended, aod::ScannedTracks>>;
using FilteredTrackData = Partition<aod::FilteredTracks>::filtered_iterator;
} // namespace aod
} // namespace o2

namespace dptdptcorrelations
{
/* all this is made configurable */
int ptbins = 18;
float ptlow = 0.2, ptup = 2.0;
int etabins = 16;
float etalow = -0.8, etaup = 0.8;
int zvtxbins = 40;
float zvtxlow = -10.0, zvtxup = 10.0;
int phibins = 72;
float philow = 0.0;
float phiup = M_PI * 2;
float phibinshift = 0.5;
float etabinwidth = (etaup - etalow) / float(etabins);
float phibinwidth = (phiup - philow) / float(phibins);
int deltaetabins = etabins * 2 - 1;
float deltaetalow = etalow - etaup, deltaetaup = etaup - etalow;
float deltaetabinwidth = (deltaetaup - deltaetalow) / float(deltaetabins);
int deltaphibins = phibins;
float deltaphibinwidth = M_PI * 2 / deltaphibins;
float deltaphilow = 0.0 - deltaphibinwidth / 2.0;
float deltaphiup = M_PI * 2 - deltaphibinwidth / 2.0;

int tracktype = 1;
int trackonecharge = 1;
int tracktwocharge = -1;
bool processpairs = false;
std::string fTaskConfigurationString = "PendingToConfigure";

/// \enum SystemType
/// \brief The type of the system under analysis
enum SystemType {
  kNoSystem = 0, ///< no system defined
  kpp,           ///< **p-p** system
  kpPb,          ///< **p-Pb** system
  kPbp,          ///< **Pb-p** system
  kPbPb,         ///< **Pb-Pb** system
  kXeXe,         ///< **Xe-Xe** system
  knSystems      ///< number of handled systems
};

/// \enum CentMultEstimatorType
/// \brief The detector used to estimate centrality/multiplicity
enum CentMultEstimatorType {
  kV0M = 0,            ///< V0M centrality/multiplicity estimator
  kV0A,                ///< V0A centrality/multiplicity estimator
  kV0C,                ///< V0C centrality/multiplicity estimator
  kCL0,                ///< CL0 centrality/multiplicity estimator
  kCL1,                ///< CL1 centrality/multiplicity estimator
  knCentMultEstimators ///< number of centrality/mutiplicity estimator
};

namespace filteranalyistask
{
//============================================================================================
// The DptDptCorrelationsFilterAnalysisTask output objects
//============================================================================================
SystemType fSystem = kNoSystem;
CentMultEstimatorType fCentMultEstimator = kV0M;
TH1F* fhCentMultB = nullptr;
TH1F* fhCentMultA = nullptr;
TH1F* fhVertexZB = nullptr;
TH1F* fhVertexZA = nullptr;
TH1F* fhPtB = nullptr;
TH1F* fhPtA = nullptr;
TH1F* fhPtPosB = nullptr;
TH1F* fhPtPosA = nullptr;
TH1F* fhPtNegB = nullptr;
TH1F* fhPtNegA = nullptr;

TH1F* fhEtaB = nullptr;
TH1F* fhEtaA = nullptr;

TH1F* fhPhiB = nullptr;
TH1F* fhPhiA = nullptr;

TH2F* fhEtaVsPhiB = nullptr;
TH2F* fhEtaVsPhiA = nullptr;

TH2F* fhPtVsEtaB = nullptr;
TH2F* fhPtVsEtaA = nullptr;
} // namespace filteranalyistask

namespace correlationstask
{

/// \enum TrackPairs
/// \brief The track combinations hadled by the class
enum TrackPairs {
  kOO = 0,    ///< one-one pairs
  kOT,        ///< one-two pairs
  kTO,        ///< two-one pairs
  kTT,        ///< two-two pairs
  nTrackPairs ///< the number of track pairs
};
} // namespace correlationstask

SystemType getSystemType()
{
  /* we have to figure out how extract the system type */
  return kPbPb;
}

template <typename CollisionObject>
bool IsEvtSelected(CollisionObject const& collision, float& centormult)
{
  using namespace filteranalyistask;

  if (collision.alias()[kINT7]) {
    if (collision.sel7()) {
      /* TODO: vertex quality checks */
      if (zvtxlow < collision.posZ() and collision.posZ() < zvtxup) {
        switch (fCentMultEstimator) {
          case kV0M:
            if (collision.centV0M() < 100 and 0 < collision.centV0M()) {
              centormult = collision.centV0M();
              return true;
            }
            break;
          default:
            break;
        }
        return false;
      }
      return false;
    }
  }
  return false;
}

template <typename TrackObject>
bool matchTrackType(TrackObject const& track)
{
  switch (tracktype) {
    case 1:
      if (track.isGlobalTrack() != 0 || track.isGlobalTrackSDD() != 0) {
        return true;
      } else {
        return false;
      }
      break;
    default:
      return false;
  }
}

template <typename TrackObject>
inline void AcceptTrack(TrackObject const& track, bool& asone, bool& astwo)
{

  asone = false;
  astwo = false;

  /* TODO: incorporate a mask in the scanned tracks table for the rejecting track reason */
  if (matchTrackType(track)) {
    if (ptlow < track.pt() and track.pt() < ptup and etalow < track.eta() and track.eta() < etaup) {
      if (((track.sign() > 0) and (trackonecharge > 0)) or ((track.sign() < 0) and (trackonecharge < 0))) {
        asone = true;
      }
      if (((track.sign() > 0) and (tracktwocharge > 0)) or ((track.sign() < 0) and (tracktwocharge < 0))) {
        astwo = true;
      }
    }
  }
}

} /* end namespace dptdptcorrelations */

// Task for <dpt,dpt> correlations analysis
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that

using namespace dptdptcorrelations;

struct DptDptCorrelationsFilterAnalysisTask {

  Configurable<int> cfgTrackType{"trktype", 1, "Type of selected tracks: 0 = no selection, 1 = global tracks FB96"};
  Configurable<bool> cfgProcessPairs{"processpairs", true, "Process pairs: false = no, just singles, true = yes, process pairs"};
  Configurable<std::string> cfgCentMultEstimator{"centmultestimator", "V0M", "Centrality/multiplicity estimator detector: default V0M"};

  Configurable<o2::analysis::DptDptBinningCuts> cfgBinning{"binning",
                                                           {28, -7.0, 7.0, 18, 0.2, 2.0, 16, -0.8, 0.8, 72, 0.5},
                                                           "triplets - nbins, min, max - for z_vtx, pT, eta and phi, binning plus bin fraction of phi origin shift"};

  OutputObj<TList> fOutput{"DptDptCorrelationsGlobalInfo", OutputObjHandlingPolicy::AnalysisObject};

  Produces<aod::AcceptedEvents> acceptedevents;
  Produces<aod::ScannedTracks> scannedtracks;

  void init(InitContext const&)
  {
    using namespace filteranalyistask;

    /* update with the configurable values */
    /* the binning */
    ptbins = cfgBinning->mPTbins;
    ptlow = cfgBinning->mPTmin;
    ptup = cfgBinning->mPTmax;
    etabins = cfgBinning->mEtabins;
    etalow = cfgBinning->mEtamin;
    etaup = cfgBinning->mEtamax;
    zvtxbins = cfgBinning->mZVtxbins;
    zvtxlow = cfgBinning->mZVtxmin;
    zvtxup = cfgBinning->mZVtxmax;
    /* the track types and combinations */
    tracktype = cfgTrackType.value;
    /* the centrality/multiplicity estimation */
    if (cfgCentMultEstimator->compare("V0M") == 0) {
      fCentMultEstimator = kV0M;
    } else {
      LOGF(FATAL, "Centrality/Multiplicity estimator %s not supported yet", cfgCentMultEstimator->c_str());
    }

    /* if the system type is not known at this time, we have to put the initalization somewhere else */
    fSystem = getSystemType();

    /* create the output list which will own the task histograms */
    TList* fOutputList = new TList();
    fOutputList->SetOwner(true);
    fOutput.setObject(fOutputList);

    /* incorporate configuration parameters to the output */
    fOutputList->Add(new TParameter<Int_t>("TrackType", cfgTrackType, 'f'));
    fOutputList->Add(new TParameter<Int_t>("TrackOneCharge", trackonecharge, 'f'));
    fOutputList->Add(new TParameter<Int_t>("TrackTwoCharge", tracktwocharge, 'f'));

    /* create the histograms */
    if (fSystem > kPbp) {
      fhCentMultB = new TH1F("CentralityB", "Centrality before cut; centrality (%)", 100, 0, 100);
      fhCentMultA = new TH1F("CentralityA", "Centrality; centrality (%)", 100, 0, 100);
    } else {
      /* for pp, pPb and Pbp systems use multiplicity instead */
      fhCentMultB = new TH1F("MultiplicityB", "Multiplicity before cut; multiplicity (%)", 100, 0, 100);
      fhCentMultA = new TH1F("MultiplicityA", "Multiplicity; multiplicity (%)", 100, 0, 100);
    }

    fhVertexZB = new TH1F("VertexZB", "Vertex Z; z_{vtx}", 60, -15, 15);
    fhVertexZA = new TH1F("VertexZA", "Vertex Z; z_{vtx}", zvtxbins, zvtxlow, zvtxup);

    fhPtB = new TH1F("fHistPtB", "p_{T} distribution for reconstructed before;p_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
    fhPtA = new TH1F("fHistPtA", "p_{T} distribution for reconstructed;p_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
    fhPtPosB = new TH1F("fHistPtPosB", "P_{T} distribution for reconstructed (#{+}) before;P_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
    fhPtPosA = new TH1F("fHistPtPosA", "P_{T} distribution for reconstructed (#{+});P_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
    fhPtNegB = new TH1F("fHistPtNegB", "P_{T} distribution for reconstructed (#{-}) before;P_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
    fhPtNegA = new TH1F("fHistPtNegA", "P_{T} distribution for reconstructed (#{-});P_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
    fhEtaB = new TH1F("fHistEtaB", "#eta distribution for reconstructed before;#eta;counts", 40, -2.0, 2.0);
    fhEtaA = new TH1F("fHistEtaA", "#eta distribution for reconstructed;#eta;counts", etabins, etalow, etaup);
    fhPhiB = new TH1F("fHistPhiB", "#phi distribution for reconstructed before;#phi;counts", 360, 0.0, 2 * M_PI);
    fhPhiA = new TH1F("fHistPhiA", "#phi distribution for reconstructed;#phi;counts", 360, 0.0, 2 * M_PI);
    fhEtaVsPhiB = new TH2F(TString::Format("CSTaskEtaVsPhiB_%s", fTaskConfigurationString.c_str()), "#eta vs #phi before;#phi;#eta", 360, 0.0, 2 * M_PI, 100, -2.0, 2.0);
    fhEtaVsPhiA = new TH2F(TString::Format("CSTaskEtaVsPhiA_%s", fTaskConfigurationString.c_str()), "#eta vs #phi;#phi;#eta", 360, 0.0, 2 * M_PI, etabins, etalow, etaup);
    fhPtVsEtaB = new TH2F(TString::Format("fhPtVsEtaB_%s", fTaskConfigurationString.c_str()), "p_{T} vs #eta before;#eta;p_{T} (GeV/c)", etabins, etalow, etaup, 100, 0.0, 15.0);
    fhPtVsEtaA = new TH2F(TString::Format("fhPtVsEtaA_%s", fTaskConfigurationString.c_str()), "p_{T} vs #eta;#eta;p_{T} (GeV/c)", etabins, etalow, etaup, ptbins, ptlow, ptup);

    /* add the hstograms to the output list */
    fOutputList->Add(fhCentMultB);
    fOutputList->Add(fhCentMultA);
    fOutputList->Add(fhVertexZB);
    fOutputList->Add(fhVertexZA);
    fOutputList->Add(fhPtB);
    fOutputList->Add(fhPtA);
    fOutputList->Add(fhPtPosB);
    fOutputList->Add(fhPtPosA);
    fOutputList->Add(fhPtNegB);
    fOutputList->Add(fhPtNegA);
    fOutputList->Add(fhEtaB);
    fOutputList->Add(fhEtaA);
    fOutputList->Add(fhPhiB);
    fOutputList->Add(fhPhiA);
    fOutputList->Add(fhEtaVsPhiB);
    fOutputList->Add(fhEtaVsPhiA);
    fOutputList->Add(fhPtVsEtaB);
    fOutputList->Add(fhPtVsEtaA);
  }

  void process(aod::CollisionEvSelCent const& collision, soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended, aod::TrackSelection> const& ftracks)
  {
    using namespace filteranalyistask;

    //    LOGF(INFO,"New collision with %d filtered tracks", ftracks.size());
    fhCentMultB->Fill(collision.centV0M());
    fhVertexZB->Fill(collision.posZ());
    bool acceptedevent = false;
    float centormult = -100.0;
    if (IsEvtSelected(collision, centormult)) {
      acceptedevent = true;
      fhCentMultA->Fill(collision.centV0M());
      fhVertexZA->Fill(collision.posZ());
      //      LOGF(INFO,"New accepted collision with %d filtered tracks", ftracks.size());

      for (auto& track : ftracks) {
        /* before track selection */
        fhPtB->Fill(track.pt());
        fhEtaB->Fill(track.eta());
        fhPhiB->Fill(track.phi());
        fhEtaVsPhiB->Fill(track.phi(), track.eta());
        fhPtVsEtaB->Fill(track.eta(), track.pt());
        if (track.sign() > 0) {
          fhPtPosB->Fill(track.pt());
        } else {
          fhPtNegB->Fill(track.pt());
        }

        /* track selection */
        /* tricky because the boolean columns issue */
        bool asone, astwo;
        AcceptTrack(track, asone, astwo);
        if (asone or astwo) {
          /* the track has been accepted */
          fhPtA->Fill(track.pt());
          fhEtaA->Fill(track.eta());
          fhPhiA->Fill(track.phi());
          fhEtaVsPhiA->Fill(track.phi(), track.eta());
          fhPtVsEtaA->Fill(track.eta(), track.pt());
          if (track.sign() > 0) {
            fhPtPosA->Fill(track.pt());
          } else {
            fhPtNegA->Fill(track.pt());
          }
        }
        scannedtracks((uint8_t)asone, (uint8_t)astwo);
      }
    } else {
      for (auto& track : ftracks) {
        scannedtracks((uint8_t) false, (uint8_t) false);
      }
    }
    acceptedevents((uint8_t)acceptedevent, centormult);
  }
};

// Task for building <dpt,dpt> correlations
struct DptDptCorrelationsTask {

  /* the data collecting engine */
  struct DataCollectingEngine {
    //============================================================================================
    // The DptDptCorrelationsAnalysisTask output objects
    //============================================================================================
    /* histograms */
    TH1F* fhN1_vsPt[2];               //!<! weighted single particle distribution vs \f$p_T\f$, track 1 and 2
    TH2F* fhN1_vsEtaPhi[2];           //!<! weighted single particle distribution vs \f$\eta,\;\phi\f$, track 1 and 2
    TH2F* fhSum1Pt_vsEtaPhi[2];       //!<! accumulated sum of weighted \f$p_T\f$ vs \f$\eta,\;\phi\f$, track 1 and 2
    TH3F* fhN1_vsZEtaPhiPt[2];        //!<! single particle distribution vs \f$\mbox{vtx}_z,\; \eta,\;\phi,\;p_T\f$, track 1 and 2
    TH3F* fhSum1Pt_vsZEtaPhiPt[2];    //!<! accumulated sum of weighted \f$p_T\f$ vs \f$\mbox{vtx}_z,\; \eta,\;\phi,\;p_T\f$, track 1 and 2
    TH2F* fhN2_vsPtPt[4];             //!<! track 1 and 2 weighted two particle distribution vs \f${p_T}_1, {p_T}_2\f$
    TH2F* fhN2_vsDEtaDPhi[4];         //!<! two-particle distribution vs \f$\Delta\eta,\;\Delta\phi\f$ 1-1,1-2,2-1,2-2, combinations
    TH2F* fhSum2PtPt_vsDEtaDPhi[4];   //!<! two-particle  \f$\sum {p_T}_1 {p_T}_2\f$ distribution vs \f$\Delta\eta,\;\Delta\phi\f$ 1-1,1-2,2-1,2-2, combinations
    TH2F* fhSum2DptDpt_vsDEtaDPhi[4]; //!<! two-particle  \f$\sum ({p_T}_1- <{p_T}_1>) ({p_T}_2 - <{p_T}_2>) \f$ distribution vs \f$\Delta\eta,\;\Delta\phi\f$ 1-1,1-2,2-1,2-2, combinations
    /* versus centrality/multiplicity  profiles */
    TProfile* fhN1_vsC[2];           //!<! weighted single particle distribution vs event centrality, track 1 and 2
    TProfile* fhSum1Pt_vsC[2];       //!<! accumulated sum of weighted \f$p_T\f$ vs event centrality, track 1 and 2
    TProfile* fhN1nw_vsC[2];         //!<! un-weighted single particle distribution vs event centrality, track 1 and 2
    TProfile* fhSum1Ptnw_vsC[2];     //!<! accumulated sum of un-weighted \f$p_T\f$ vs event centrality, track 1 and 2
    TProfile* fhN2_vsC[4];           //!<! weighted accumulated two particle distribution vs event centrality 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2PtPt_vsC[4];     //!<! weighted accumulated \f${p_T}_1 {p_T}_2\f$ distribution vs event centrality 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2DptDpt_vsC[4];   //!<! weighted accumulated \f$\sum ({p_T}_1- <{p_T}_1>) ({p_T}_2 - <{p_T}_2>) \f$ distribution vs event centrality 1-1,1-2,2-1,2-2, combinations
    TProfile* fhN2nw_vsC[4];         //!<! un-weighted accumulated two particle distribution vs event centrality 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2PtPtnw_vsC[4];   //!<! un-weighted accumulated \f${p_T}_1 {p_T}_2\f$ distribution vs event centrality 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2DptDptnw_vsC[4]; //!<! un-weighted accumulated \f$\sum ({p_T}_1- <{p_T}_1>) ({p_T}_2 - <{p_T}_2>) \f$ distribution vs \f$\Delta\eta,\;\Delta\phi\f$ distribution vs event centrality 1-1,1-2,2-1,2-2, combinations

    const char* tname[2] = {"1", "2"}; ///< the external track names, one and two, for histogram creation
    const char* trackPairsNames[4] = {"OO", "OT", "TO", "TT"};

    /// \brief Returns the potentially phi origin shifted phi
    /// \param phi the track azimuthal angle
    /// \return the track phi origin shifted azimuthal angle
    float GetShiftedPhi(float phi)
    {
      if (not(phi < phiup)) {
        return phi - M_PI * 2;
      } else {
        return phi;
      }
    }

    /// \brief Returns the zero based bin index of the eta phi passed track
    /// \param t the intended track
    /// \return the zero based bin index
    ///
    /// According to the bining structure, to the track eta will correspond
    /// a zero based bin index and similarlly for the track phi
    /// The returned index is the composition of both considering eta as
    /// the first index component
    /// WARNING: for performance reasons no checks are done about the consistency
    /// of track's eta and phin with the corresponding ranges so, it is suppossed
    /// the track has been accepted and it is within that ranges
    /// IF THAT IS NOT THE CASE THE ROUTINE WILL PRODUCE NONSENSE RESULTS
    template <typename TrackObject>
    int GetEtaPhiIndex(TrackObject const& t)
    {
      int etaix = int((t.eta() - etalow) / etabinwidth);
      /* consider a potential phi origin shift */
      float phi = GetShiftedPhi(t.phi());
      int phiix = int((phi - philow) / phibinwidth);
      return etaix * phibins + phiix;
    }

    /// \brief Returns the TH2 global index for the differential histograms
    /// \param t1 the intended track one
    /// \param t2 the intended track two
    /// \return the globl TH2 bin for delta eta delta phi
    ///
    /// WARNING: for performance reasons no checks are done about the consistency
    /// of tracks' eta and phi within the corresponding ranges so, it is suppossed
    /// the tracks have been accepted and they are within that ranges
    /// IF THAT IS NOT THE CASE THE ROUTINE WILL PRODUCE NONSENSE RESULTS
    template <typename TrackObject>
    int GetDEtaDPhiGlobalIndex(TrackObject const& t1, TrackObject const& t2)
    {
      using namespace correlationstask;

      /* rule: ix are always zero based while bins are always one based */
      int etaix_1 = int((t1.eta() - etalow) / etabinwidth);
      /* consider a potential phi origin shift */
      float phi = GetShiftedPhi(t1.phi());
      int phiix_1 = int((phi - philow) / phibinwidth);
      int etaix_2 = int((t2.eta() - etalow) / etabinwidth);
      /* consider a potential phi origin shift */
      phi = GetShiftedPhi(t2.phi());
      int phiix_2 = int((phi - philow) / phibinwidth);

      int deltaeta_ix = etaix_1 - etaix_2 + etabins - 1;
      int deltaphi_ix = phiix_1 - phiix_2;
      if (deltaphi_ix < 0) {
        deltaphi_ix += phibins;
      }

      return fhN2_vsDEtaDPhi[kOO]->GetBin(deltaeta_ix + 1, deltaphi_ix + 1);
    }

    /// \brief fills the singles histograms in singles execution mode
    /// \param passedtracks filtered table with the tracks associated to the passed index
    /// \param tix index, in the singles histogram bank, for the passed filetered track table
    template <typename TrackListObject>
    void processSingles(TrackListObject const& passedtracks, int tix, float zvtx)
    {
      for (auto& track : passedtracks) {
        double corr = 1.0; /* TODO: track correction  weights */
        fhN1_vsPt[tix]->Fill(track.pt(), corr);
        fhN1_vsZEtaPhiPt[tix]->Fill(zvtx, GetEtaPhiIndex(track) + 0.5, track.pt(), corr);
        fhSum1Pt_vsZEtaPhiPt[tix]->Fill(zvtx, GetEtaPhiIndex(track) + 0.5, track.pt(), corr);
      }
    }

    /// \brief fills the singles histograms in pair execution mode
    /// \param passedtracks filtered table with the tracks associated to the passed index
    /// \param tix index, in the singles histogram bank, for the passed filetered track table
    /// \param cmul centrality - multiplicity for the collision being analyzed
    template <typename TrackListObject>
    void processTracks(TrackListObject const& passedtracks, int tix, float cmul)
    {
      /* process magnitudes */
      double n1 = 0;       ///< weighted number of track 1 tracks for current collision
      double sum1Pt = 0;   ///< accumulated sum of weighted track 1 \f$p_T\f$ for current collision
      double n1nw = 0;     ///< not weighted number of track 1 tracks for current collision
      double sum1Ptnw = 0; ///< accumulated sum of not weighted track 1 \f$p_T\f$ for current collision
      for (auto& track : passedtracks) {
        double corr = 1.0; /* TODO: track correction  weights */
        n1 += corr;
        sum1Pt += track.pt() * corr;
        n1nw += 1;
        sum1Ptnw += track.pt();

        fhN1_vsEtaPhi[tix]->Fill(track.eta(), GetShiftedPhi(track.phi()), corr);
        fhSum1Pt_vsEtaPhi[tix]->Fill(track.eta(), GetShiftedPhi(track.phi()), track.pt() * corr);
      }
      fhN1_vsC[tix]->Fill(cmul, n1);
      fhSum1Pt_vsC[tix]->Fill(cmul, sum1Pt);
      fhN1nw_vsC[tix]->Fill(cmul, n1nw);
      fhSum1Ptnw_vsC[tix]->Fill(cmul, sum1Ptnw);
    }

    /// \brief fills the pair histograms in pair execution mode
    /// \param trks1 filtered table with the tracks associated to the first track in the pair
    /// \param trks2 filtered table with the tracks associated to the second track in the pair
    /// \param pix index, in the track combination histogram bank, for the passed filetered track tables
    /// \param cmul centrality - multiplicity for the collision being analyzed
    /// Be aware that at least in half of the cases traks1 and trks2 will have the same content
    template <typename TrackListObject>
    void processTrackPairs(TrackListObject const& trks1, TrackListObject const& trks2, int pix, float cmul)
    {
      /* process pair magnitudes */
      double n2 = 0;           ///< weighted number of track 1 track 2 pairs for current collision
      double sum2PtPt = 0;     ///< accumulated sum of weighted track 1 track 2 \f${p_T}_1 {p_T}_2\f$ for current collision
      double sum2DptDpt = 0;   ///< accumulated sum of weighted number of track 1 tracks times weighted track 2 \f$p_T\f$ for current collision
      double n2nw = 0;         ///< not weighted number of track1 track 2 pairs for current collision
      double sum2PtPtnw = 0;   ///< accumulated sum of not weighted track 1 track 2 \f${p_T}_1 {p_T}_2\f$ for current collision
      double sum2DptDptnw = 0; ///< accumulated sum of not weighted number of track 1 tracks times not weighted track 2 \f$p_T\f$ for current collision
      for (auto& track1 : trks1) {
        double ptavg_1 = 0.0; /* TODO: load ptavg_1 for eta1, phi1 bin */
        double corr1 = 1.0;   /* TODO: track correction  weights */
        for (auto& track2 : trks2) {
          /* checkiing the same track id condition */
          if (track1 == track2) {
            /* exclude autocorrelations */
            continue;
          } else {
            /* process pair magnitudes */
            double ptavg_2 = 0.0; /* TODO: load ptavg_2 for eta2, phi2 bin */
            double corr2 = 1.0;   /* TODO: track correction  weights */
            double corr = corr1 * corr2;
            double dptdpt = (track1.pt() - ptavg_1) * (track2.pt() - ptavg_2);
            n2 += corr;
            sum2PtPt += track1.pt() * track2.pt() * corr;
            sum2DptDpt += corr * dptdpt;
            n2nw += 1;
            sum2PtPtnw += track1.pt() * track2.pt();
            sum2DptDptnw += dptdpt;
            /* get the global bin for filling the differential histograms */
            int globalbin = GetDEtaDPhiGlobalIndex(track1, track2);
            fhN2_vsDEtaDPhi[pix]->AddBinContent(globalbin, corr);
            fhSum2DptDpt_vsDEtaDPhi[pix]->AddBinContent(globalbin, corr * dptdpt);
            fhSum2PtPt_vsDEtaDPhi[pix]->AddBinContent(globalbin, track1.pt() * track2.pt() * corr);
            fhN2_vsPtPt[pix]->Fill(track1.pt(), track2.pt(), corr);
          }
        }
      }
      fhN2_vsC[pix]->Fill(cmul, n2);
      fhSum2PtPt_vsC[pix]->Fill(cmul, sum2PtPt);
      fhSum2DptDpt_vsC[pix]->Fill(cmul, sum2DptDpt);
      fhN2nw_vsC[pix]->Fill(cmul, n2nw);
      fhSum2PtPtnw_vsC[pix]->Fill(cmul, sum2PtPtnw);
      fhSum2DptDptnw_vsC[pix]->Fill(cmul, sum2DptDptnw);
      /* let's also update the number of entries in the differential histograms */
      fhN2_vsDEtaDPhi[pix]->SetEntries(fhN2_vsDEtaDPhi[pix]->GetEntries() + n2);
      fhSum2DptDpt_vsDEtaDPhi[pix]->SetEntries(fhSum2DptDpt_vsDEtaDPhi[pix]->GetEntries() + n2);
      fhSum2PtPt_vsDEtaDPhi[pix]->SetEntries(fhSum2PtPt_vsDEtaDPhi[pix]->GetEntries() + n2);
    }

    void init(TList* fOutputList)
    {
      using namespace correlationstask;

      /* create the histograms */
      Bool_t oldstatus = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE);

      if (!processpairs) {
        for (int i = 0; i < 2; ++i) {
          /* histograms for each track, one and two */
          fhN1_vsPt[i] = new TH1F(TString::Format("n1_%s_vsPt", tname[i]).Data(),
                                  TString::Format("#LT n_{1} #GT;p_{t,%s} (GeV/c);#LT n_{1} #GT", tname[i]).Data(),
                                  ptbins, ptlow, ptup);
          /* we don't want the Sumw2 structure being created here */
          bool defSumw2 = TH1::GetDefaultSumw2();
          TH1::SetDefaultSumw2(false);
          fhN1_vsZEtaPhiPt[i] = new TH3F(TString::Format("n1_%s_vsZ_vsEtaPhi_vsPt", tname[i]).Data(),
                                         TString::Format("#LT n_{1} #GT;vtx_{z};#eta_{%s}#times#varphi_{%s};p_{t,%s} (GeV/c)", tname[i], tname[i], tname[i]).Data(),
                                         zvtxbins, zvtxlow, zvtxup, etabins * phibins, 0.0, double(etabins * phibins), ptbins, ptlow, ptup);
          fhSum1Pt_vsZEtaPhiPt[i] = new TH3F(TString::Format("sumPt1_%s_vsZ_vsEtaPhi_vsPt", tname[i]).Data(),
                                             TString::Format("#LT #Sigma p_{t,%s}#GT;vtx_{z};#eta_{%s}#times#varphi_{%s};p_{t,%s} (GeV/c)", tname[i], tname[i], tname[i], tname[i]).Data(),
                                             zvtxbins, zvtxlow, zvtxup, etabins * phibins, 0.0, double(etabins * phibins), ptbins, ptlow, ptup);
          /* we return it back to previuos state */
          TH1::SetDefaultSumw2(defSumw2);

          /* the statistical uncertainties will be estimated by the subsamples method so let's get rid of the error tracking */
          fhN1_vsZEtaPhiPt[i]->SetBit(TH1::kIsNotW);
          fhN1_vsZEtaPhiPt[i]->Sumw2(false);
          fhSum1Pt_vsZEtaPhiPt[i]->SetBit(TH1::kIsNotW);
          fhSum1Pt_vsZEtaPhiPt[i]->Sumw2(false);

          fOutputList->Add(fhN1_vsPt[i]);
          fOutputList->Add(fhN1_vsZEtaPhiPt[i]);
          fOutputList->Add(fhSum1Pt_vsZEtaPhiPt[i]);
        }
      } else {
        for (int i = 0; i < 2; ++i) {
          /* histograms for each track, one and two */
          fhN1_vsEtaPhi[i] = new TH2F(TString::Format("n1_%s_vsEtaPhi", tname[i]).Data(),
                                      TString::Format("#LT n_{1} #GT;#eta_{%s};#varphi_{%s} (radian);#LT n_{1} #GT", tname[i], tname[i]).Data(),
                                      etabins, etalow, etaup, phibins, philow, phiup);
          fhSum1Pt_vsEtaPhi[i] = new TH2F(TString::Format("sumPt_%s_vsEtaPhi", tname[i]).Data(),
                                          TString::Format("#LT #Sigma p_{t,%s} #GT;#eta_{%s};#varphi_{%s} (radian);#LT #Sigma p_{t,%s} #GT (GeV/c)",
                                                          tname[i], tname[i], tname[i], tname[i])
                                            .Data(),
                                          etabins, etalow, etaup, phibins, philow, phiup);
          fhN1_vsC[i] = new TProfile(TString::Format("n1_%s_vsM", tname[i]).Data(),
                                     TString::Format("#LT n_{1} #GT (weighted);Centrality (%%);#LT n_{1} #GT").Data(),
                                     100, 0.0, 100.0);
          fhSum1Pt_vsC[i] = new TProfile(TString::Format("sumPt_%s_vsM", tname[i]),
                                         TString::Format("#LT #Sigma p_{t,%s} #GT (weighted);Centrality (%%);#LT #Sigma p_{t,%s} #GT (GeV/c)", tname[i], tname[i]).Data(),
                                         100, 0.0, 100.0);
          fhN1nw_vsC[i] = new TProfile(TString::Format("n1Nw_%s_vsM", tname[i]).Data(),
                                       TString::Format("#LT n_{1} #GT;Centrality (%%);#LT n_{1} #GT").Data(),
                                       100, 0.0, 100.0);
          fhSum1Ptnw_vsC[i] = new TProfile(TString::Format("sumPtNw_%s_vsM", tname[i]).Data(),
                                           TString::Format("#LT #Sigma p_{t,%s} #GT;Centrality (%%);#LT #Sigma p_{t,%s} #GT (GeV/c)", tname[i], tname[i]).Data(), 100, 0.0, 100.0);
          fOutputList->Add(fhN1_vsEtaPhi[i]);
          fOutputList->Add(fhSum1Pt_vsEtaPhi[i]);
          fOutputList->Add(fhN1_vsC[i]);
          fOutputList->Add(fhSum1Pt_vsC[i]);
          fOutputList->Add(fhN1nw_vsC[i]);
          fOutputList->Add(fhSum1Ptnw_vsC[i]);
        }

        for (int i = 0; i < nTrackPairs; ++i) {
          /* histograms for each track pair combination */
          /* we don't want the Sumw2 structure being created here */
          bool defSumw2 = TH1::GetDefaultSumw2();
          TH1::SetDefaultSumw2(false);
          const char* pname = trackPairsNames[i];
          fhN2_vsDEtaDPhi[i] = new TH2F(TString::Format("n2_12_vsDEtaDPhi_%s", pname), TString::Format("#LT n_{2} #GT (%s);#Delta#eta;#Delta#varphi;#LT n_{2} #GT", pname),
                                        deltaetabins, deltaetalow, deltaetaup, deltaphibins, deltaphilow, deltaphiup);
          fhSum2PtPt_vsDEtaDPhi[i] = new TH2F(TString::Format("sumPtPt_12_vsDEtaDPhi_%s", pname), TString::Format("#LT #Sigma p_{t,1}p_{t,2} #GT (%s);#Delta#eta;#Delta#varphi;#LT #Sigma p_{t,1}p_{t,2} #GT (GeV^{2})", pname),
                                              deltaetabins, deltaetalow, deltaetaup, deltaphibins, deltaphilow, deltaphiup);
          fhSum2DptDpt_vsDEtaDPhi[i] = new TH2F(TString::Format("sumDptDpt_12_vsDEtaDPhi_%s", pname), TString::Format("#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (%s);#Delta#eta;#Delta#varphi;#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (GeV^{2})", pname),
                                                deltaetabins, deltaetalow, deltaetaup, deltaphibins, deltaphilow, deltaphiup);
          /* we return it back to previuos state */
          TH1::SetDefaultSumw2(defSumw2);

          fhN2_vsPtPt[i] = new TH2F(TString::Format("n2_12_vsPtVsPt_%s", pname), TString::Format("#LT n_{2} #GT (%s);p_{t,1} (GeV/c);p_{t,2} (GeV/c);#LT n_{2} #GT", pname),
                                    ptbins, ptlow, ptup, ptbins, ptlow, ptup);

          fhN2_vsC[i] = new TProfile(TString::Format("n2_12_vsM_%s", pname), TString::Format("#LT n_{2} #GT (%s) (weighted);Centrality (%%);#LT n_{2} #GT", pname), 100, 0.0, 100.0);
          fhSum2PtPt_vsC[i] = new TProfile(TString::Format("sumPtPt_12_vsM_%s", pname), TString::Format("#LT #Sigma p_{t,1}p_{t,2} #GT (%s) (weighted);Centrality (%%);#LT #Sigma p_{t,1}p_{t,2} #GT (GeV^{2})", pname), 100, 0.0, 100.0);
          fhSum2DptDpt_vsC[i] = new TProfile(TString::Format("sumDptDpt_12_vsM_%s", pname), TString::Format("#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (%s) (weighted);Centrality (%%);#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (GeV^{2})", pname), 100, 0.0, 100.0);
          fhN2nw_vsC[i] = new TProfile(TString::Format("n2Nw_12_vsM_%s", pname), TString::Format("#LT n_{2} #GT (%s);Centrality (%%);#LT n_{2} #GT", pname), 100, 0.0, 100.0);
          fhSum2PtPtnw_vsC[i] = new TProfile(TString::Format("sumPtPtNw_12_vsM_%s", pname), TString::Format("#LT #Sigma p_{t,1}p_{t,2} #GT (%s);Centrality (%%);#LT #Sigma p_{t,1}p_{t,2} #GT (GeV^{2})", pname), 100, 0.0, 100.0);
          fhSum2DptDptnw_vsC[i] = new TProfile(TString::Format("sumDptDptNw_12_vsM_%s", pname), TString::Format("#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (%s);Centrality (%%);#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (GeV^{2})", pname), 100, 0.0, 100.0);

          /* the statistical uncertainties will be estimated by the subsamples method so let's get rid of the error tracking */
          fhN2_vsDEtaDPhi[i]->SetBit(TH1::kIsNotW);
          fhN2_vsDEtaDPhi[i]->Sumw2(false);
          fhSum2PtPt_vsDEtaDPhi[i]->SetBit(TH1::kIsNotW);
          fhSum2PtPt_vsDEtaDPhi[i]->Sumw2(false);
          fhSum2DptDpt_vsDEtaDPhi[i]->SetBit(TH1::kIsNotW);
          fhSum2DptDpt_vsDEtaDPhi[i]->Sumw2(false);

          fOutputList->Add(fhN2_vsDEtaDPhi[i]);
          fOutputList->Add(fhSum2PtPt_vsDEtaDPhi[i]);
          fOutputList->Add(fhSum2DptDpt_vsDEtaDPhi[i]);
          fOutputList->Add(fhN2_vsPtPt[i]);
          fOutputList->Add(fhN2_vsC[i]);
          fOutputList->Add(fhSum2PtPt_vsC[i]);
          fOutputList->Add(fhSum2DptDpt_vsC[i]);
          fOutputList->Add(fhN2nw_vsC[i]);
          fOutputList->Add(fhSum2PtPtnw_vsC[i]);
          fOutputList->Add(fhSum2DptDptnw_vsC[i]);
        }
      }
      TH1::AddDirectory(oldstatus);
    }
  }; // DataCollectingEngine

  /* the data memebers for this task */
  /* the centrality / multiplicity limits for collecting data in this task instance */
  int ncmranges = 0;
  float* fCentMultMin = nullptr;
  float* fCentMultMax = nullptr;

  /* the data collecting engine instances */
  DataCollectingEngine** dataCE;

  Configurable<bool> cfgProcessPairs{"processpairs", false, "Process pairs: false = no, just singles, true = yes, process pairs"};
  Configurable<std::string> cfgCentSpec{"centralities", "00-05,05-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80", "Centrality/multiplicity ranges in min-max separated by commas"};

  Configurable<o2::analysis::DptDptBinningCuts> cfgBinning{"binning",
                                                           {28, -7.0, 7.0, 18, 0.2, 2.0, 16, -0.8, 0.8, 72, 0.5},
                                                           "triplets - nbins, min, max - for z_vtx, pT, eta and phi, binning plus bin fraction of phi origin shift"};

  OutputObj<TList> fOutput{"DptDptCorrelationsData", OutputObjHandlingPolicy::AnalysisObject};

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == (uint8_t) true);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true) or (aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true));

  Partition<aod::FilteredTracks> Tracks1 = aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true;
  Partition<aod::FilteredTracks> Tracks2 = aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true;

  void init(InitContext const&)
  {
    using namespace correlationstask;

    /* update with the configurable values */
    ptbins = cfgBinning->mPTbins;
    ptlow = cfgBinning->mPTmin;
    ptup = cfgBinning->mPTmax;
    etabins = cfgBinning->mEtabins;
    etalow = cfgBinning->mEtamin;
    etaup = cfgBinning->mEtamax;
    zvtxbins = cfgBinning->mZVtxbins;
    zvtxlow = cfgBinning->mZVtxmin;
    zvtxup = cfgBinning->mZVtxmax;
    phibins = cfgBinning->mPhibins;
    philow = 0.0f;
    phiup = M_PI * 2;
    phibinshift = cfgBinning->mPhibinshift;
    processpairs = cfgProcessPairs.value;
    /* update the potential binning change */
    etabinwidth = (etaup - etalow) / float(etabins);
    phibinwidth = (phiup - philow) / float(phibins);

    /* the differential bining */
    deltaetabins = etabins * 2 - 1;
    deltaetalow = etalow - etaup, deltaetaup = etaup - etalow;
    deltaetabinwidth = (deltaetaup - deltaetalow) / float(deltaetabins);
    deltaphibins = phibins;
    deltaphibinwidth = M_PI * 2 / deltaphibins;
    deltaphilow = 0.0 - deltaphibinwidth / 2.0;
    deltaphiup = M_PI * 2 - deltaphibinwidth / 2.0;

    /* create the output directory which will own the task output */
    TList* fGlobalOutputList = new TList();
    fGlobalOutputList->SetName("CorrelationsDataReco");
    fGlobalOutputList->SetOwner(true);
    fOutput.setObject(fGlobalOutputList);

    /* incorporate configuration parameters to the output */
    fGlobalOutputList->Add(new TParameter<Int_t>("NoBinsVertexZ", zvtxbins, 'f'));
    fGlobalOutputList->Add(new TParameter<Int_t>("NoBinsPt", ptbins, 'f'));
    fGlobalOutputList->Add(new TParameter<Int_t>("NoBinsEta", etabins, 'f'));
    fGlobalOutputList->Add(new TParameter<Int_t>("NoBinsPhi", phibins, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MinVertexZ", zvtxlow, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MaxVertexZ", zvtxup, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MinPt", ptlow, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MaxPt", ptup, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MinEta", etalow, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MaxEta", etaup, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MinPhi", philow, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("MaxPhi", phiup, 'f'));
    fGlobalOutputList->Add(new TParameter<Double_t>("PhiBinShift", phibinshift, 'f'));
    fGlobalOutputList->Add(new TParameter<Bool_t>("DifferentialOutput", true, 'f'));

    /* after the parameters dump the proper phi limits are set according to the phi shift */
    phiup = phiup - phibinwidth * phibinshift;
    philow = philow - phibinwidth * phibinshift;

    /* create the data collecting engine instances according to the configured centrality/multiplicity ranges */
    {
      TObjArray* tokens = TString(cfgCentSpec.value.c_str()).Tokenize(",");
      ncmranges = tokens->GetEntries();
      fCentMultMin = new float[ncmranges];
      fCentMultMax = new float[ncmranges];
      dataCE = new DataCollectingEngine*[ncmranges];

      for (int i = 0; i < ncmranges; ++i) {
        float cmmin = 0.0f;
        float cmmax = 0.0f;
        sscanf(tokens->At(i)->GetName(), "%f-%f", &cmmin, &cmmax);
        fCentMultMin[i] = cmmin;
        fCentMultMax[i] = cmmax;
        dataCE[i] = new DataCollectingEngine();

        /* crete the output list for the current centrality range */
        TList* fOutputList = new TList();
        fOutputList->SetName(TString::Format("DptDptCorrelationsData-%s", tokens->At(i)->GetName()));
        fOutputList->SetOwner(true);
        /* init the data collection instance */
        dataCE[i]->init(fOutputList);
        fGlobalOutputList->Add(fOutputList);
      }
      delete tokens;
      for (int i = 0; i < ncmranges; ++i) {
        LOGF(INFO, " centrality/multipliicty range: %d, low limit: %f, up limit: %f", i, fCentMultMin[i], fCentMultMax[i]);
      }
    }
  }

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents, aod::AcceptedEvents>>::iterator const& collision, aod::FilteredTracks const& tracks)
  {
    using namespace correlationstask;

    /* locate the data collecting engine for the collision centrality/multiplicity */
    int ixDCE = 0;
    float cm = collision.centmult();
    bool rgfound = false;
    for (int i = 0; i < ncmranges; ++i) {
      if (cm < fCentMultMax[i]) {
        rgfound = true;
        ixDCE = i;
        break;
      }
    }

    if (rgfound) {
      if (not processpairs) {
        /* process single tracks */
        dataCE[ixDCE]->processSingles(Tracks1, 0, collision.posZ()); /* track one */
        dataCE[ixDCE]->processSingles(Tracks2, 1, collision.posZ()); /* track two */
      } else {
        /* process track magnitudes */
        /* TODO: the centrality should be chosen non detector dependent */
        dataCE[ixDCE]->processTracks(Tracks1, 0, collision.centmult()); /* track one */
        dataCE[ixDCE]->processTracks(Tracks2, 1, collision.centmult()); /* track one */
        /* process pair magnitudes */
        dataCE[ixDCE]->processTrackPairs(Tracks1, Tracks1, kOO, collision.centmult());
        dataCE[ixDCE]->processTrackPairs(Tracks1, Tracks2, kOT, collision.centmult());
        dataCE[ixDCE]->processTrackPairs(Tracks2, Tracks1, kTO, collision.centmult());
        dataCE[ixDCE]->processTrackPairs(Tracks2, Tracks2, kTT, collision.centmult());
      }
    }
  }
};

// Task for building <dpt,dpt> correlations
struct TracksAndEventClassificationQA {

  Configurable<o2::analysis::SimpleInclusiveCut> cfg{"mycfg", {"mycfg", 3, 2.0f}, "A Configurable Object, default mycfg.x=3, mycfg.y=2.0"};

  OutputObj<TH1F> fTracksOne{TH1F("TracksOne", "Tracks as track one;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fTracksTwo{TH1F("TracksTwo", "Tracks as track two;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fTracksOneAndTwo{TH1F("TracksOneAndTwo", "Tracks as track one and as track two;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fTracksNone{TH1F("TracksNone", "Not selected tracks;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fTracksOneUnsel{TH1F("TracksOneUnsel", "Tracks as track one;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fTracksTwoUnsel{TH1F("TracksTwoUnsel", "Tracks as track two;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fTracksOneAndTwoUnsel{TH1F("TracksOneAndTwoUnsel", "Tracks as track one and as track two;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fTracksNoneUnsel{TH1F("TracksNoneUnsel", "Not selected tracks;number of tracks;events", 1500, 0.0, 1500.0)};
  OutputObj<TH1F> fSelectedEvents{TH1F("SelectedEvents", "Selected events;;events", 2, 0.0, 2.0)};

  void init(InitContext const&)
  {
    fSelectedEvents->GetXaxis()->SetBinLabel(1, "Not selected events");
    fSelectedEvents->GetXaxis()->SetBinLabel(2, "Selected events");
  }

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == (uint8_t) true);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true) or (aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true));

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents, aod::AcceptedEvents>>::iterator const& collision,
               soa::Filtered<soa::Join<aod::Tracks, aod::ScannedTracks>> const& tracks)
  {
    if (collision.eventaccepted() != (uint8_t) true) {
      fSelectedEvents->Fill(0.5);
    } else {
      fSelectedEvents->Fill(1.5);
    }

    int ntracks_one = 0;
    int ntracks_two = 0;
    int ntracks_one_and_two = 0;
    int ntracks_none = 0;
    for (auto& track : tracks) {
      if ((track.trackacceptedasone() != (uint8_t) true) and (track.trackacceptedastwo() != (uint8_t) true)) {
        ntracks_none++;
      }
      if ((track.trackacceptedasone() == (uint8_t) true) and (track.trackacceptedastwo() == (uint8_t) true)) {
        ntracks_one_and_two++;
      }
      if (track.trackacceptedasone() == (uint8_t) true) {
        ntracks_one++;
      }
      if (track.trackacceptedastwo() == (uint8_t) true) {
        ntracks_two++;
      }
    }
    if (collision.eventaccepted() != (uint8_t) true) {
      /* control for non selected events */
      fTracksOneUnsel->Fill(ntracks_one);
      fTracksTwoUnsel->Fill(ntracks_two);
      fTracksNoneUnsel->Fill(ntracks_none);
      fTracksOneAndTwoUnsel->Fill(ntracks_one_and_two);
    } else {
      fTracksOne->Fill(ntracks_one);
      fTracksTwo->Fill(ntracks_two);
      fTracksNone->Fill(ntracks_none);
      fTracksOneAndTwo->Fill(ntracks_one_and_two);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<DptDptCorrelationsFilterAnalysisTask>(cfgc),
    adaptAnalysisTask<TracksAndEventClassificationQA>(cfgc),
    adaptAnalysisTask<DptDptCorrelationsTask>(cfgc)};
  return workflow;
}
