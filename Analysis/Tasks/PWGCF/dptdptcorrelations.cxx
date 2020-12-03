// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec centspec = {"centralities",
                              VariantType::String,
                              "00-05,05-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80",
                              {"Centrality/multiplicity ranges in min-max separated by commas"}};
  workflowOptions.push_back(centspec);
}

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
float phiup = TMath::TwoPi();
float phibinshift = 0.5;
float etabinwidth = (etaup - etalow) / float(etabins);
float phibinwidth = (phiup - philow) / float(phibins);
int tracktype = 1;
int trackonecharge = 1;
int tracktwocharge = -1;
bool processpairs = false;
std::string fTaskConfigurationString = "PendingToConfigure";

/* while we don't have proper bool columnst */
uint8_t DPTDPT_TRUE = 1;
uint8_t DPTDPT_FALSE = 0;

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
//============================================================================================
// The DptDptCorrelationsAnalysisTask output objects
//============================================================================================
/* histograms */
TH1F* fhN1_1_vsPt;            //!<! track 1 weighted single particle distribution vs \f$p_T\f$
TH2F* fhN1_1_vsEtaPhi;        //!<! track 1 weighted single particle distribution vs \f$\eta,\;\phi\f$
TH2F* fhSum1Pt_1_vsEtaPhi;    //!<! track 1 accumulated sum of weighted \f$p_T\f$ vs \f$\eta,\;\phi\f$
TH3F* fhN1_1_vsZEtaPhiPt;     //!<! track 1 single particle distribution vs \f$\mbox{vtx}_z,\; \eta,\;\phi,\;p_T\f$
TH1F* fhN1_2_vsPt;            //!<! track 2 weighted single particle distribution vs \f$p_T\f$
TH2F* fhN1_2_vsEtaPhi;        //!<! track 2 weighted single particle distribution vs \f$\eta,\;\phi\f$
TH2F* fhSum1Pt_2_vsEtaPhi;    //!<! track 2 accumulated sum of weighted \f$p_T\f$ vs \f$\eta,\;\phi\f$
TH3F* fhN1_2_vsZEtaPhiPt;     //!<! track 2 single particle distribution vs \f$\mbox{vtx}_z,\;\eta,\;\phi,\;p_T\f$
TH2F* fhN2_12_vsPtPt;         //!<! track 1 and 2 weighted two particle distribution vs \f${p_T}_1, {p_T}_2\f$
TH1F* fhN2_12_vsEtaPhi;       //!<! track 1 and 2 weighted two particle distribution vs \f$\eta,\;\phi\f$
TH1F* fhSum2PtPt_12_vsEtaPhi; //!<! track 1 and 2 weighted accumulated \f${p_T}_1 {p_T}_2\f$ distribution vs \f$\eta,\;\phi\f$
TH1F* fhSum2PtN_12_vsEtaPhi;  //!<! track 1 and 2 weighted accumulated \f${p_T}_1 n_2\f$ distribution vs \f$\eta,\;\phi\f$
TH1F* fhSum2NPt_12_vsEtaPhi;  //!<! track 1 and 2 weighted accumulated \f$n_1 {p_T}_2\f$ distribution vs \f$\eta,\;\phi\f$
/* versus centrality/multiplicity  profiles */
TProfile* fhN1_1_vsC;          //!<! track 1 weighted single particle distribution vs event centrality
TProfile* fhSum1Pt_1_vsC;      //!<! track 1 accumulated sum of weighted \f$p_T\f$ vs event centrality
TProfile* fhN1nw_1_vsC;        //!<! track 1 un-weighted single particle distribution vs event centrality
TProfile* fhSum1Ptnw_1_vsC;    //!<! track 1 accumulated sum of un-weighted \f$p_T\f$ vs event centrality
TProfile* fhN1_2_vsC;          //!<! track 2 weighted single particle distribution vs event centrality
TProfile* fhSum1Pt_2_vsC;      //!<! track 2 accumulated sum of weighted \f$p_T\f$ vs event centrality
TProfile* fhN1nw_2_vsC;        //!<! track 2 un-weighted single particle distribution vs event centrality
TProfile* fhSum1Ptnw_2_vsC;    //!<! track 2 accumulated sum of un-weighted \f$p_T\f$ vs event centrality
TProfile* fhN2_12_vsC;         //!<! track 1 and 2 weighted two particle distribution vs event centrality
TProfile* fhSum2PtPt_12_vsC;   //!<! track 1 and 2 weighted accumulated \f${p_T}_1 {p_T}_2\f$ distribution vs event centrality
TProfile* fhSum2PtN_12_vsC;    //!<! track 1 and 2 weighted accumulated \f${p_T}_1 n_2\f$ distribution vs event centrality
TProfile* fhSum2NPt_12_vsC;    //!<! track 1 and 2 weighted accumulated \f$n_1 {p_T}_2\f$ distribution vs event centrality
TProfile* fhN2nw_12_vsC;       //!<! track 1 and 2 un-weighted two particle distribution vs event centrality
TProfile* fhSum2PtPtnw_12_vsC; //!<! track 1 and 2 un-weighted accumulated \f${p_T}_1 {p_T}_2\f$ distribution vs event centrality
TProfile* fhSum2PtNnw_12_vsC;  //!<! track 1 and 2 un-weighted accumulated \f${p_T}_1 n_2\f$ distribution vs event centrality
TProfile* fhSum2NPtnw_12_vsC;  //!<! track 1 and 2 un-weighted accumulated \f$n_1 {p_T}_2\f$ distribution vs event centrality
} // namespace correlationstask

SystemType getSystemType()
{
  /* we have to figure out how extract the system type */
  return kPbPb;
}

bool IsEvtSelected(aod::CollisionEvSelCent const& collision, float& centormult)
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

bool matchTrackType(aod::TrackData const& track)
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

inline void AcceptTrack(aod::TrackData const& track, uint8_t& asone, uint8_t& astwo)
{

  asone = DPTDPT_FALSE;
  astwo = DPTDPT_FALSE;

  /* TODO: incorporate a mask in the scanned tracks table for the rejecting track reason */
  if (matchTrackType(track)) {
    if (ptlow < track.pt() and track.pt() < ptup and etalow < track.eta() and track.eta() < etaup) {
      if (((track.charge() > 0) and (trackonecharge > 0)) or ((track.charge() < 0) and (trackonecharge < 0))) {
        asone = DPTDPT_TRUE;
      }
      if (((track.charge() > 0) and (tracktwocharge > 0)) or ((track.charge() < 0) and (tracktwocharge < 0))) {
        astwo = DPTDPT_TRUE;
      }
    }
  }
}

/// \brief Returns the potentially phi origin shifted phi
/// \param phi the track azimuthal angle
/// \return the track phi origin shifted azimuthal angle
inline float GetShiftedPhi(float phi)
{
  if (not(phi < phiup)) {
    return phi - TMath::TwoPi();
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
inline int GetEtaPhiIndex(aod::FilteredTrackData const& t)
{
  int etaix = int((t.eta() - etalow) / etabinwidth);
  /* consider a potential phi origin shift */
  float phi = GetShiftedPhi(t.phi());
  int phiix = int((phi - philow) / phibinwidth);
  return etaix * phibins + phiix;
}
} /* end namespace dptdptcorrelations */

// Task for <dpt,dpt> correlations analysis
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that

using namespace dptdptcorrelations;

struct DptDptCorrelationsFilterAnalysisTask {

  Configurable<int> cfgTrackType{"trktype", 1, "Type of selected tracks: 0 = no selection, 1 = global tracks FB96"};
  Configurable<int> cfgTrackOneCharge{"trk1charge", -1, "Trakc one charge: 1 = positive, -1 = negative"};
  Configurable<int> cfgTrackTwoCharge{"trk2charge", -1, "Trakc two charge: 1 = positive, -1 = negative"};
  Configurable<bool> cfgProcessPairs{"processpairs", true, "Process pairs: false = no, just singles, true = yes, process pairs"};
  Configurable<std::string> cfgCentMultEstimator{"centmultestimator", "V0M", "Centrality/multiplicity estimator detector: default V0M"};

  Configurable<o2::analysis::DptDptBinningCuts> cfgBinning{"binning",
                                                           {28, -7.0, 7.0, 18, 0.2, 2.0, 16, -0.8, 0.8, 72, 0.5},
                                                           "triplets - nbins, min, max - for z_vtx, pT, eta and phi, binning plus bin fraction of phi origin shift"};

  OutputObj<TList> fOutput{"DptDptCorrelationsGlobalInfo", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TFolder> fConfOutput{"DptDptCorrelationsGlobalConfiguration", OutputObjHandlingPolicy::AnalysisObject};

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
    trackonecharge = cfgTrackOneCharge.value;
    tracktwocharge = cfgTrackTwoCharge.value;
    /* the centrality/multiplicity estimation */
    if (cfgCentMultEstimator->compare("V0M") == 0) {
      fCentMultEstimator = kV0M;
    } else {
      LOGF(FATAL, "Centrality/Multiplicity estimator %s not supported yet", cfgCentMultEstimator->c_str());
    }

    /* if the system type is not known at this time, we have to put the initalization somewhere else */
    fSystem = getSystemType();

    /* create the configuration folder which will own the task configuration parameters */
    TFolder* fOutputFolder = new TFolder("DptDptCorrelationsGlobalConfigurationFolder", "DptDptCorrelationsGlobalConfigurationFolder");
    fOutputFolder->SetOwner(true);
    fConfOutput.setObject(fOutputFolder);

    /* incorporate configuration parameters to the output */
    fOutputFolder->Add(new TParameter<Int_t>("TrackType", cfgTrackType, 'f'));
    fOutputFolder->Add(new TParameter<Int_t>("TrackOneCharge", cfgTrackOneCharge, 'f'));
    fOutputFolder->Add(new TParameter<Int_t>("TrackTwoCharge", cfgTrackTwoCharge, 'f'));

    /* create the output list which will own the task histograms */
    TList* fOutputList = new TList();
    fOutputList->SetOwner(true);
    fOutput.setObject(fOutputList);

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
    int acceptedevent = DPTDPT_FALSE;
    float centormult = -100.0;
    if (IsEvtSelected(collision, centormult)) {
      acceptedevent = DPTDPT_TRUE;
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
        if (track.charge() > 0) {
          fhPtPosB->Fill(track.pt());
        } else {
          fhPtNegB->Fill(track.pt());
        }

        /* track selection */
        /* tricky because the boolean columns issue */
        uint8_t asone, astwo;
        AcceptTrack(track, asone, astwo);
        if ((asone == DPTDPT_TRUE) or (astwo == DPTDPT_TRUE)) {
          /* the track has been accepted */
          fhPtA->Fill(track.pt());
          fhEtaA->Fill(track.eta());
          fhPhiA->Fill(track.phi());
          fhEtaVsPhiA->Fill(track.phi(), track.eta());
          fhPtVsEtaA->Fill(track.eta(), track.pt());
          if (track.charge() > 0) {
            fhPtPosA->Fill(track.pt());
          } else {
            fhPtNegA->Fill(track.pt());
          }
        }
        scannedtracks(asone, astwo);
      }
    } else {
      for (auto& track : ftracks) {
        scannedtracks(DPTDPT_FALSE, DPTDPT_FALSE);
      }
    }
    acceptedevents(acceptedevent, centormult);
  }
};

// Task for building <dpt,dpt> correlations
struct DptDptCorrelationsTask {
  /* the data memebers for this task */
  /* the centrality / multiplicity limits for collecting data in this task instance */
  float fCentMultMin;
  float fCentMultMax;

  Configurable<bool> cfgProcessPairs{"processpairs", false, "Process pairs: false = no, just singles, true = yes, process pairs"};

  Configurable<o2::analysis::DptDptBinningCuts> cfgBinning{"binning",
                                                           {28, -7.0, 7.0, 18, 0.2, 2.0, 16, -0.8, 0.8, 72, 0.5},
                                                           "triplets - nbins, min, max - for z_vtx, pT, eta and phi, binning plus bin fraction of phi origin shift"};

  OutputObj<TList> fOutput{"DptDptCorrelationsData", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TFolder> fConfOutput{"DptDptCorrelationsConfiguration", OutputObjHandlingPolicy::AnalysisObject};

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == DPTDPT_TRUE);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE) or (aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE));

  Partition<aod::FilteredTracks> Tracks1 = aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE;
  Partition<aod::FilteredTracks> Tracks2 = aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE;

  DptDptCorrelationsTask(float cmmin,
                         float cmmax,
                         Configurable<bool> _cfgProcessPairs = {"processpairs", false, "Process pairs: false = no, just singles, true = yes, process pairs"},
                         Configurable<o2::analysis::DptDptBinningCuts> _cfgBinning = {"binning",
                                                                                      {28, -7.0, 7.0, 18, 0.2, 2.0, 16, -0.8, 0.8, 72, 0.5},
                                                                                      "triplets - nbins, min, max - for z_vtx, pT, eta and phi, binning plus bin fraction of phi origin shift"},
                         OutputObj<TList> _fOutput = {"DptDptCorrelationsData", OutputObjHandlingPolicy::AnalysisObject},
                         OutputObj<TFolder> _fConfOutput = {"DptDptCorrelationsConfiguration", OutputObjHandlingPolicy::AnalysisObject},
                         Filter _onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == DPTDPT_TRUE),
                         Filter _onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE) or (aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE)),
                         Partition<aod::FilteredTracks> _Tracks1 = aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE,
                         Partition<aod::FilteredTracks> _Tracks2 = aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE)
    : fCentMultMin(cmmin),
      fCentMultMax(cmmax),
      cfgProcessPairs(_cfgProcessPairs),
      cfgBinning(_cfgBinning),
      fOutput(_fOutput),
      fConfOutput(_fConfOutput),
      onlyacceptedevents((aod::dptdptcorrelations::eventaccepted == DPTDPT_TRUE) and (aod::dptdptcorrelations::centmult > fCentMultMin) and (aod::dptdptcorrelations::centmult < fCentMultMax)),
      onlyacceptedtracks((aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE) or (aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE)),
      Tracks1(aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE),
      Tracks2(aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE)
  {
  }

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
    phiup = TMath::TwoPi();
    phibinshift = cfgBinning->mPhibinshift;
    processpairs = cfgProcessPairs.value;
    /* update the potential binning change */
    etabinwidth = (etaup - etalow) / float(etabins);
    phibinwidth = (phiup - philow) / float(phibins);

    /* create the configuration folder which will own the task configuration parameters */
    TFolder* fOutputFolder = new TFolder("DptDptCorrelationsConfigurationFolder", "DptDptCorrelationsConfigurationFolder");
    fOutputFolder->SetOwner(true);
    fConfOutput.setObject(fOutputFolder);

    /* incorporate configuration parameters to the output */
    fOutputFolder->Add(new TParameter<Int_t>("NoBinsVertexZ", zvtxbins, 'f'));
    fOutputFolder->Add(new TParameter<Int_t>("NoBinsPt", ptbins, 'f'));
    fOutputFolder->Add(new TParameter<Int_t>("NoBinsEta", etabins, 'f'));
    fOutputFolder->Add(new TParameter<Int_t>("NoBinsPhi", phibins, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MinVertexZ", zvtxlow, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MaxVertexZ", zvtxup, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MinPt", ptlow, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MaxPt", ptup, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MinEta", etalow, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MaxEta", etaup, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MinPhi", philow, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("MaxPhi", phiup, 'f'));
    fOutputFolder->Add(new TParameter<Double_t>("PhiBinShift", phibinshift, 'f'));

    /* after the parameters dump the proper phi limits are set according to the phi shift */
    phiup = phiup - phibinwidth * phibinshift;
    philow = philow - phibinwidth * phibinshift;

    /* create the output list which will own the task histograms */
    TList* fOutputList = new TList();
    fOutputList->SetOwner(true);
    fOutput.setObject(fOutputList);

    /* create the histograms */
    Bool_t oldstatus = TH1::AddDirectoryStatus();
    TH1::AddDirectory(kFALSE);

    if (!processpairs) {
      fhN1_1_vsPt = new TH1F("n1_1_vsPt", "#LT n_{1} #GT;p_{t,1} (GeV/c);#LT n_{1} #GT", ptbins, ptlow, ptup);
      fhN1_2_vsPt = new TH1F("n1_2_vsPt", "#LT n_{1} #GT;p_{t,2} (GeV/c);#LT n_{1} #GT", ptbins, ptlow, ptup);
      /* we don't want the Sumw2 structure being created here */
      bool defSumw2 = TH1::GetDefaultSumw2();
      TH1::SetDefaultSumw2(false);
      fhN1_1_vsZEtaPhiPt = new TH3F("n1_1_vsZ_vsEtaPhi_vsPt", "#LT n_{1} #GT;vtx_{z};#eta_{1}#times#varphi_{1};p_{t,1} (GeV/c)",
                                    zvtxbins, zvtxlow, zvtxup, etabins * phibins, 0.0, double(etabins * phibins), ptbins, ptlow, ptup);

      fhN1_2_vsZEtaPhiPt = new TH3F("n1_2_vsZ_vsEtaPhi_vsPt", "#LT n_{2} #GT;vtx_{z};#eta_{2}#times#varphi_{2};p_{t,2} (GeV/c)",
                                    zvtxbins, zvtxlow, zvtxup, etabins * phibins, 0.0, double(etabins * phibins), ptbins, ptlow, ptup);
      /* we return it back to previuos state */
      TH1::SetDefaultSumw2(defSumw2);

      /* the statistical uncertainties will be estimated by the subsamples method so let's get rid of the error tracking */
      fhN1_1_vsZEtaPhiPt->SetBit(TH1::kIsNotW);
      fhN1_1_vsZEtaPhiPt->Sumw2(false);
      fhN1_2_vsZEtaPhiPt->SetBit(TH1::kIsNotW);
      fhN1_2_vsZEtaPhiPt->Sumw2(false);

      fOutputList->Add(fhN1_1_vsPt);
      fOutputList->Add(fhN1_1_vsZEtaPhiPt);
      fOutputList->Add(fhN1_2_vsPt);
      fOutputList->Add(fhN1_2_vsZEtaPhiPt);
    } else {
      fhN1_1_vsEtaPhi = new TH2F("n1_1_vsEtaPhi", "#LT n_{1} #GT;#eta_{1};#varphi_{1} (radian);#LT n_{1} #GT",
                                 etabins, etalow, etaup, phibins, philow, phiup);
      fhSum1Pt_1_vsEtaPhi = new TH2F("sumPt_1_vsEtaPhi", "#LT #Sigma p_{t,1} #GT;#eta_{1};#varphi_{1} (radian);#LT #Sigma p_{t,1} #GT (GeV/c)",
                                     etabins, etalow, etaup, phibins, philow, phiup);
      fhN1_1_vsC = new TProfile("n1_1_vsM", "#LT n_{1} #GT (weighted);Centrality (%);#LT n_{1} #GT", 100, 0.0, 100.0);
      fhSum1Pt_1_vsC = new TProfile("sumPt_1_vsM", "#LT #Sigma p_{t,1} #GT (weighted);Centrality (%);#LT #Sigma p_{t,1} #GT (GeV/c)", 100, 0.0, 100.0);
      fhN1nw_1_vsC = new TProfile("n1Nw_1_vsM", "#LT n_{1} #GT;Centrality (%);#LT n_{1} #GT", 100, 0.0, 100.0);
      fhSum1Ptnw_1_vsC = new TProfile("sumPtNw_1_vsM", "#LT #Sigma p_{t,1} #GT;Centrality (%);#LT #Sigma p_{t,1} #GT (GeV/c)", 100, 0.0, 100.0);

      fhN1_2_vsEtaPhi = new TH2F("n1_2_vsEtaPhi", "#LT n_{1} #GT;#eta_{2};#varphi_{2} (radian);#LT n_{1} #GT",
                                 etabins, etalow, etaup, phibins, philow, phiup);
      fhSum1Pt_2_vsEtaPhi = new TH2F("sumPt_2_vsEtaPhi", "#LT #Sigma p_{t,2} #GT;#eta_{2};#varphi_{2} (radian);#LT #Sigma p_{t,2} #GT (GeV/c)",
                                     etabins, etalow, etaup, phibins, philow, phiup);
      fhN1_2_vsC = new TProfile("n1_2_vsM", "#LT n_{1} #GT (weighted);Centrality (%);#LT n_{1} #GT", 100, 0.0, 100.0);
      fhSum1Pt_2_vsC = new TProfile("sumPt_2_vsM", "#LT #Sigma p_{t,1} #GT (weighted);Centrality (%);#LT #Sigma p_{t,1} #GT (GeV/c)", 100, 0.0, 100.0);
      fhN1nw_2_vsC = new TProfile("n1Nw_2_vsM", "#LT n_{1} #GT;Centrality (%);#LT n_{1} #GT", 100, 0.0, 100.0);
      fhSum1Ptnw_2_vsC = new TProfile("sumPtNw_2_vsM", "#LT #Sigma p_{t,1} #GT;Centrality (%);#LT #Sigma p_{t,1} #GT (GeV/c)", 100, 0.0, 100.0);

      /* we don't want the Sumw2 structure being created here */
      bool defSumw2 = TH1::GetDefaultSumw2();
      TH1::SetDefaultSumw2(false);
      fhN2_12_vsEtaPhi = new TH1F("n2_12_vsEtaPhi", "#LT n_{2} #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT n_{2} #GT",
                                  etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
      fhSum2PtPt_12_vsEtaPhi = new TH1F("sumPtPt_12_vsEtaPhi", "#LT #Sigma p_{t,1}p_{t,2} #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT #Sigma p_{t,1}p_{t,2} #GT (GeV)^{2}",
                                        etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
      fhSum2PtN_12_vsEtaPhi = new TH1F("sumPtN_12_vsEtaPhi", "#LT #Sigma p_{t,1}N #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT #Sigma p_{t,1}N #GT (GeV)",
                                       etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
      fhSum2NPt_12_vsEtaPhi = new TH1F("sumNPt_12_vsEtaPhi", "#LT N#Sigma p_{t,2} #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT N#Sigma p_{t,2} #GT (GeV)",
                                       etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
      /* we return it back to previuos state */
      TH1::SetDefaultSumw2(defSumw2);

      fhN2_12_vsPtPt = new TH2F("n2_12_vsPtVsPt", "#LT n_{2} #GT;p_{t,1} (GeV/c);p_{t,2} (GeV/c);#LT n_{2} #GT",
                                ptbins, ptlow, ptup, ptbins, ptlow, ptup);
      fhN2_12_vsC = new TProfile("n2_12_vsM", "#LT n_{2} #GT (weighted);Centrality (%);#LT n_{2} #GT", 100, 0.0, 100.0);
      fhSum2PtPt_12_vsC = new TProfile("sumPtPt_12_vsM", "#LT #Sigma p_{t,1}p_{t,2} #GT (weighted);Centrality (%);#LT #Sigma p_{t,1}p_{t,2} #GT (GeV)^{2}", 100, 0.0, 100.0);
      fhSum2PtN_12_vsC = new TProfile("sumPtN_12_vsM", "#LT #Sigma p_{t,1}N #GT (weighted);Centrality (%);#LT #Sigma p_{t,1}N #GT (GeV)", 100, 0.0, 100.0);
      fhSum2NPt_12_vsC = new TProfile("sumNPt_12_vsM", "#LT N#Sigma p_{t,2} #GT (weighted);Centrality (%);#LT N#Sigma p_{t,2} #GT (GeV)", 100, 0.0, 100.0);
      fhN2nw_12_vsC = new TProfile("n2Nw_12_vsM", "#LT n_{2} #GT;Centrality (%);#LT n_{2} #GT", 100, 0.0, 100.0);
      fhSum2PtPtnw_12_vsC = new TProfile("sumPtPtNw_12_vsM", "#LT #Sigma p_{t,1}p_{t,2} #GT;Centrality (%);#LT #Sigma p_{t,1}p_{t,2} #GT (GeV)^{2}", 100, 0.0, 100.0);
      fhSum2PtNnw_12_vsC = new TProfile("sumPtNNw_12_vsM", "#LT #Sigma p_{t,1}N #GT;Centrality (%);#LT #Sigma p_{t,1}N #GT (GeV)", 100, 0.0, 100.0);
      fhSum2NPtnw_12_vsC = new TProfile("sumNPtNw_12_vsM", "#LT N#Sigma p_{t,2} #GT;Centrality (%);#LT N#Sigma p_{t,2} #GT (GeV)", 100, 0.0, 100.0);

      /* the statistical uncertainties will be estimated by the subsamples method so let's get rid of the error tracking */
      fhN2_12_vsEtaPhi->SetBit(TH1::kIsNotW);
      fhN2_12_vsEtaPhi->Sumw2(false);
      fhSum2PtPt_12_vsEtaPhi->SetBit(TH1::kIsNotW);
      fhSum2PtPt_12_vsEtaPhi->Sumw2(false);
      fhSum2PtN_12_vsEtaPhi->SetBit(TH1::kIsNotW);
      fhSum2PtN_12_vsEtaPhi->Sumw2(false);
      fhSum2NPt_12_vsEtaPhi->SetBit(TH1::kIsNotW);
      fhSum2NPt_12_vsEtaPhi->Sumw2(false);

      fOutputList->Add(fhN1_1_vsEtaPhi);
      fOutputList->Add(fhSum1Pt_1_vsEtaPhi);
      fOutputList->Add(fhN1_1_vsC);
      fOutputList->Add(fhSum1Pt_1_vsC);
      fOutputList->Add(fhN1nw_1_vsC);
      fOutputList->Add(fhSum1Ptnw_1_vsC);
      fOutputList->Add(fhN1_2_vsEtaPhi);
      fOutputList->Add(fhSum1Pt_2_vsEtaPhi);
      fOutputList->Add(fhN1_2_vsC);
      fOutputList->Add(fhSum1Pt_2_vsC);
      fOutputList->Add(fhN1nw_2_vsC);
      fOutputList->Add(fhSum1Ptnw_2_vsC);

      fOutputList->Add(fhN2_12_vsEtaPhi);
      fOutputList->Add(fhSum2PtPt_12_vsEtaPhi);
      fOutputList->Add(fhSum2PtN_12_vsEtaPhi);
      fOutputList->Add(fhSum2NPt_12_vsEtaPhi);
      fOutputList->Add(fhN2_12_vsPtPt);
      fOutputList->Add(fhN2_12_vsC);
      fOutputList->Add(fhSum2PtPt_12_vsC);
      fOutputList->Add(fhSum2PtN_12_vsC);
      fOutputList->Add(fhSum2NPt_12_vsC);
      fOutputList->Add(fhN2nw_12_vsC);
      fOutputList->Add(fhSum2PtPtnw_12_vsC);
      fOutputList->Add(fhSum2PtNnw_12_vsC);
      fOutputList->Add(fhSum2NPtnw_12_vsC);
    }

    TH1::AddDirectory(oldstatus);
  }

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents, aod::AcceptedEvents>>::iterator const& collision,
               aod::FilteredTracks const& tracks)
  {
    using namespace correlationstask;

    if (not processpairs) {
      /* process single tracks */
      for (auto& track1 : Tracks1) {
        double corr = 1.0; /* TODO: track correction  weights */
        fhN1_1_vsPt->Fill(track1.pt(), corr);
        fhN1_1_vsZEtaPhiPt->Fill(collision.posZ(), GetEtaPhiIndex(track1) + 0.5, track1.pt(), corr);
      }
      for (auto& track2 : Tracks2) {
        double corr = 1.0; /* TODO: track correction  weights */
        fhN1_2_vsPt->Fill(track2.pt(), corr);
        fhN1_2_vsZEtaPhiPt->Fill(collision.posZ(), GetEtaPhiIndex(track2) + 0.5, track2.pt(), corr);
      }
    } else {
      {
        /* process track one magnitudes */
        double n1_1 = 0;       ///< weighted number of track 1 tracks for current collision
        double sum1Pt_1 = 0;   ///< accumulated sum of weighted track 1 \f$p_T\f$ for current collision
        double n1nw_1 = 0;     ///< not weighted number of track 1 tracks for current collision
        double sum1Ptnw_1 = 0; ///< accumulated sum of not weighted track 1 \f$p_T\f$ for current collision
        for (auto& track1 : Tracks1) {
          double corr = 1.0; /* TODO: track correction  weights */
          n1_1 += corr;
          sum1Pt_1 += track1.pt() * corr;
          n1nw_1 += 1;
          sum1Ptnw_1 += track1.pt();

          fhN1_1_vsEtaPhi->Fill(track1.eta(), GetShiftedPhi(track1.phi()), corr);
          fhSum1Pt_1_vsEtaPhi->Fill(track1.eta(), GetShiftedPhi(track1.phi()), track1.pt() * corr);
        }
        /* TODO: the centrality should be chosen non detector dependent */
        fhN1_1_vsC->Fill(collision.centmult(), n1_1);
        fhSum1Pt_1_vsC->Fill(collision.centmult(), sum1Pt_1);
        fhN1nw_1_vsC->Fill(collision.centmult(), n1nw_1);
        fhSum1Ptnw_1_vsC->Fill(collision.centmult(), sum1Ptnw_1);
      }
      {
        /* process track two magnitudes */
        double n1_2 = 0;       ///< weighted number of track 2 tracks for current collisiont
        double sum1Pt_2 = 0;   ///< accumulated sum of weighted track 2 \f$p_T\f$ for current collision
        double n1nw_2 = 0;     ///< not weighted number of track 2 tracks for current collision
        double sum1Ptnw_2 = 0; ///< accumulated sum of not weighted track 2 \f$p_T\f$ for current collision
        for (auto& track2 : Tracks2) {
          double corr = 1.0; /* TODO: track correction  weights */
          n1_2 += corr;
          sum1Pt_2 += track2.pt() * corr;
          n1nw_2 += 1;
          sum1Ptnw_2 += track2.pt();

          fhN1_2_vsEtaPhi->Fill(track2.eta(), GetShiftedPhi(track2.phi()), corr);
          fhSum1Pt_2_vsEtaPhi->Fill(track2.eta(), GetShiftedPhi(track2.phi()), track2.pt() * corr);
        }
        /* TODO: the centrality should be chosen non detector dependent */
        fhN1_2_vsC->Fill(collision.centmult(), n1_2);
        fhSum1Pt_2_vsC->Fill(collision.centmult(), sum1Pt_2);
        fhN1nw_2_vsC->Fill(collision.centmult(), n1nw_2);
        fhSum1Ptnw_2_vsC->Fill(collision.centmult(), sum1Ptnw_2);
      }
      /* process pairs of tracks */
      // this will be desireable
      //      for (auto& [track1, track2] : combinations(CombinationsFullIndexPolicy(Tracks1, Tracks2))) {
      /* process pair magnitudes */
      double n2_12 = 0;         ///< weighted number of track 1 track 2 pairs for current collision
      double sum2PtPt_12 = 0;   ///< accumulated sum of weighted track 1 track 2 \f${p_T}_1 {p_T}_2\f$ for current collision
      double sum2NPt_12 = 0;    ///< accumulated sum of weighted number of track 1 tracks times weighted track 2 \f$p_T\f$ for current collision
      double sum2PtN_12 = 0;    ///< accumulated sum of weighted track 1 \f$p_T\f$ times weighted number of track 2 tracks for current collision
      double n2nw_12 = 0;       ///< not weighted number of track1 track 2 pairs for current collision
      double sum2PtPtnw_12 = 0; ///< accumulated sum of not weighted track 1 track 2 \f${p_T}_1 {p_T}_2\f$ for current collision
      double sum2NPtnw_12 = 0;  ///< accumulated sum of not weighted number of track 1 tracks times not weighted track 2 \f$p_T\f$ for current collision
      double sum2PtNnw_12 = 0;  ///< accumulated sum of not weighted track 1 \f$p_T\f$ times not weighted number of track  tracks for current collision
      for (auto& track1 : Tracks1) {
        for (auto& track2 : Tracks2) {
          /* checkiing the same track id condition */
          if (track1.index() == track2.index()) {
            /* exclude autocorrelations */
          } else {
            /* process pair magnitudes */
            double corr1 = 1.0; /* TODO: track correction  weights */
            double corr2 = 1.0; /* TODO: track correction  weights */
            double corr = corr1 * corr2;
            n2_12 += corr;
            sum2PtPt_12 += track1.pt() * track2.pt() * corr;
            sum2NPt_12 += corr * track2.pt();
            sum2PtN_12 += track1.pt() * corr;
            n2nw_12 += 1;
            sum2PtPtnw_12 += track1.pt() * track2.pt();
            sum2NPtnw_12 += track2.pt();
            sum2PtNnw_12 += track1.pt();
            /* we already know the bin in the flattened histograms, let's use it to update them */
            fhN2_12_vsEtaPhi->AddBinContent(GetEtaPhiIndex(track1) * etabins * phibins + GetEtaPhiIndex(track2) + 1, corr);
            fhSum2NPt_12_vsEtaPhi->AddBinContent(GetEtaPhiIndex(track1) * etabins * phibins + GetEtaPhiIndex(track2) + 1, corr * track2.pt());
            fhSum2PtN_12_vsEtaPhi->AddBinContent(GetEtaPhiIndex(track1) * etabins * phibins + GetEtaPhiIndex(track2) + 1, track1.pt() * corr);
            fhSum2PtPt_12_vsEtaPhi->AddBinContent(GetEtaPhiIndex(track1) * etabins * phibins + GetEtaPhiIndex(track2) + 1, track1.pt() * track2.pt() * corr);
            fhN2_12_vsPtPt->Fill(track1.pt(), track2.pt(), corr);
          }
        }
      }
      fhN2_12_vsC->Fill(collision.centmult(), n2_12);
      fhSum2PtPt_12_vsC->Fill(collision.centmult(), sum2PtPt_12);
      fhSum2PtN_12_vsC->Fill(collision.centmult(), sum2PtN_12);
      fhSum2NPt_12_vsC->Fill(collision.centmult(), sum2NPt_12);
      fhN2nw_12_vsC->Fill(collision.centmult(), n2nw_12);
      fhSum2PtPtnw_12_vsC->Fill(collision.centmult(), sum2PtPtnw_12);
      fhSum2PtNnw_12_vsC->Fill(collision.centmult(), sum2PtNnw_12);
      fhSum2NPtnw_12_vsC->Fill(collision.centmult(), sum2NPtnw_12);
      /* let's also update the number of entries in the flattened histograms */
      fhN2_12_vsEtaPhi->SetEntries(fhN2_12_vsEtaPhi->GetEntries() + n2nw_12);
      fhSum2NPt_12_vsEtaPhi->SetEntries(fhSum2NPt_12_vsEtaPhi->GetEntries() + n2nw_12);
      fhSum2PtN_12_vsEtaPhi->SetEntries(fhSum2PtN_12_vsEtaPhi->GetEntries() + n2nw_12);
      fhSum2PtPt_12_vsEtaPhi->SetEntries(fhSum2PtPt_12_vsEtaPhi->GetEntries() + n2nw_12);
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

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == DPTDPT_TRUE);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE) or (aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE));

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents, aod::AcceptedEvents>>::iterator const& collision,
               soa::Filtered<soa::Join<aod::Tracks, aod::ScannedTracks>> const& tracks)
  {
    if (collision.eventaccepted() != DPTDPT_TRUE) {
      fSelectedEvents->Fill(0.5);
    } else {
      fSelectedEvents->Fill(1.5);
    }

    int ntracks_one = 0;
    int ntracks_two = 0;
    int ntracks_one_and_two = 0;
    int ntracks_none = 0;
    for (auto& track : tracks) {
      if ((track.trackacceptedasone() != DPTDPT_TRUE) and (track.trackacceptedastwo() != DPTDPT_TRUE)) {
        ntracks_none++;
      }
      if ((track.trackacceptedasone() == DPTDPT_TRUE) and (track.trackacceptedastwo() == DPTDPT_TRUE)) {
        ntracks_one_and_two++;
      }
      if (track.trackacceptedasone() == DPTDPT_TRUE) {
        ntracks_one++;
      }
      if (track.trackacceptedastwo() == DPTDPT_TRUE) {
        ntracks_two++;
      }
    }
    if (collision.eventaccepted() != DPTDPT_TRUE) {
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
  std::string myoptcentralities = cfgc.options().get<std::string>("centralities");
  TObjArray* tokens = TString(myoptcentralities.c_str()).Tokenize(",");
  int nranges = tokens->GetEntries();

  WorkflowSpec workflow{
    adaptAnalysisTask<DptDptCorrelationsFilterAnalysisTask>("DptDptCorrelationsFilterAnalysisTask"),
    adaptAnalysisTask<TracksAndEventClassificationQA>("TracksAndEventClassificationQA")};
  for (int i = 0; i < nranges; ++i) {
    float cmmin = 0.0f;
    float cmmax = 0.0f;
    sscanf(tokens->At(i)->GetName(), "%f-%f", &cmmin, &cmmax);
    workflow.push_back(adaptAnalysisTask<DptDptCorrelationsTask>(Form("DptDptCorrelationsTask-%s", tokens->At(i)->GetName()), cmmin, cmmax));
  }
  delete tokens;
  return workflow;
}
