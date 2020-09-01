// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Centrality.h"
#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"
#include <TROOT.h>
#include <TParameter.h>
#include <TList.h>
#include <TDirectory.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TProfile3D.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::soa;
using namespace o2::framework::expressions;

namespace o2
{
namespace aod
{
/* we have to change from int to bool when bool columns work properly */
namespace dptdptcorrelations
{
DECLARE_SOA_COLUMN(EventAccepted, eventaccepted, uint8_t);
DECLARE_SOA_COLUMN(TrackacceptedAsOne, trackacceptedasone, uint8_t);
DECLARE_SOA_COLUMN(TrackacceptedAsTwo, trackacceptedastwo, uint8_t);
} // namespace dptdptcorrelations
DECLARE_SOA_TABLE(AcceptedEvents, "AOD", "ACCEPTEDEVENTS", dptdptcorrelations::EventAccepted);
DECLARE_SOA_TABLE(ScannedTracks, "AOD", "SCANNEDTRACKS", dptdptcorrelations::TrackacceptedAsOne, dptdptcorrelations::TrackacceptedAsTwo);

using CollisionEvSelCent = soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator;
using TrackData = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended, aod::TrackSelection>::iterator;
using FilteredTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended, aod::ScannedTracks>>;
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
/* this still panding of being configurable */
int phibins = 72;

float philow = 0.0;
float phiup = TMath::TwoPi();
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

//============================================================================================
// The DptDptCorrelationsFilterAnalysisTask output objects
//============================================================================================
SystemType fSystem = kNoSystem;
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

SystemType getSystemType()
{
  /* we have to figure out how extract the system type */
  return kPbPb;
}

bool IsEvtSelected(aod::CollisionEvSelCent const& collision)
{
  if (collision.alias()[kINT7]) {
    if (collision.sel7()) {
      if (zvtxlow < collision.posZ() and collision.posZ() < zvtxup) {
        return true;
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
      if (track.isGlobalTrack() != 0)
        return true;
      else
        return false;
      break;
    case 2:
      if (track.isGlobalTrackSDD() != 0)
        return true;
      else
        return false;
      break;
    default:
      return false;
  }
}

inline std::tuple<uint8_t, uint8_t> AcceptTrack(aod::TrackData const& track)
{

  uint8_t asone = DPTDPT_FALSE;
  uint8_t astwo = DPTDPT_FALSE;

  /* TODO: incorporate a mask in the scanned tracks table for the rejecting track reason */
  if (matchTrackType(track)) {
    if (ptlow < track.pt() and track.pt() < ptup and etalow < track.eta() and track.eta() < etaup) {
      if (((track.charge() > 0) and (trackonecharge > 0)) or ((track.charge() < 0) and (trackonecharge < 0))) {
        asone = DPTDPT_TRUE;
      } else if (((track.charge() > 0) and (tracktwocharge > 0)) or ((track.charge() < 0) and (tracktwocharge < 0))) {
        astwo = DPTDPT_TRUE;
      }
    }
  }
  return std::make_tuple(asone, astwo);
}

/// \brief Returns the zero based bin index of the eta phi passed values
/// \param eta the eta value
/// \param phi the phi value
/// \return the zero based bin index
///
/// According to the bining structure, to the track eta will correspond
/// a zero based bin index and similarlly for the track phi
/// The returned index is the composition of both considering eta as
/// the first index component
inline int GetEtaPhiIndex(float eta, float phi)
{
  int etaix = int((eta - etalow) / etabinwidth);
  int phiix = int((phi - philow) / phibinwidth);
  return etaix * phibins + phiix;
}
} /* end namespace dptdptcorrelations */

// Task for <dpt,dpt> correlations analysis
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that

using namespace dptdptcorrelations;

struct DptDptCorrelationsFilterAnalysisTask {

  Configurable<int> cfgTrackType{"trktype", 1, "Type of selected tracks: 0 = no selection, 1 = global tracks loose DCA, 2 = global SDD tracks"};
  Configurable<int> cfgTrackOneCharge{"trk1charge", 1, "Trakc one charge: 1 = positive, -1 = negative"};
  Configurable<int> cfgTrackTwoCharge{"trk2charge", -1, "Trakc two charge: 1 = positive, -1 = negative"};

  Configurable<int> cfgPtBins{"ptbins", 18, "Number of pT bins, default 18"};
  Configurable<int> cfgZVtxBins{"zvtxbins", 28, "Number of z_vtx bins, default 28"};
  Configurable<int> cfgEtaBins{"etabins", 16, "Number of eta bins, default 16"};
  Configurable<float> cfgPtLow{"ptlow", 0.2f, "Lowest pT (GeV/c), default 0.2"};
  Configurable<float> cfgPtHigh{"pthigh", 2.0f, "Highest pT (GeV/c), default 2.0"};
  Configurable<float> cfgZVtxLow{"zvtxlow", -7.0f, "Lowest z_vtx distance (cm), default -7.0"};
  Configurable<float> cfgZVtxHigh{"zvtxhigh", 7.0f, "Highest z_vtx distance (cm), default 7.0"};
  Configurable<float> cfgEtaLow{"etalow", -0.8f, "Lowest eta value, default -0.8"};
  Configurable<float> cfgEtaHigh{"etahigh", 0.8f, "Highest eta value, default 0.8"};

  OutputObj<TList> fOutput{"DptDptCorrelationsGlobalInfo", OutputObjHandlingPolicy::AnalysisObject};

  Produces<aod::AcceptedEvents> acceptedevents;
  Produces<aod::ScannedTracks> scannedtracks;

  void init(InitContext const&)
  {
    /* update the configurable values */
    ptbins = cfgPtBins.value;
    ptlow = cfgPtLow.value;
    ptup = cfgPtHigh.value;
    etabins = cfgEtaBins.value;
    etalow = cfgEtaLow.value;
    etaup = cfgEtaHigh.value;
    zvtxbins = cfgZVtxBins.value;
    zvtxlow = cfgZVtxLow.value;
    zvtxup = cfgZVtxHigh.value;
    tracktype = cfgTrackType.value;
    trackonecharge = cfgTrackOneCharge.value;
    tracktwocharge = cfgTrackTwoCharge.value;

    /* still missing configuration */
    phibins = 72;

    /* if the system type is not known at this time, we have to put the initalization somewhere else */
    fSystem = getSystemType();

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
    fhEtaB = new TH1F("fHistEtaB", "#eta distribution for reconstructed before;#eta;counts", 40, 2.0, 2.0);
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
    //    LOGF(INFO,"New collision with %d filtered tracks", ftracks.size());
    fhCentMultB->Fill(collision.centV0M());
    fhVertexZB->Fill(collision.posZ());
    int acceptedevent = DPTDPT_FALSE;
    if (IsEvtSelected(collision)) {
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
        auto [asone, astwo] = AcceptTrack(track);
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
    acceptedevents(acceptedevent);
  }
};

// Task for building <dpt,dpt> correlations
struct DptDptCorrelationsTask {

  Configurable<int> cfgTrackType{"trktype", 1, "Type of selected tracks: 0 = no selection, 1 = global tracks loose DCA, 2 = global SDD tracks"};
  Configurable<int> cfgTrackOneCharge{"trk1charge", 1, "Trakc one charge: 1 = positive, -1 = negative"};
  Configurable<int> cfgTrackTwoCharge{"trk2charge", -1, "Trakc two charge: 1 = positive, -1 = negative"};
  Configurable<bool> cfgProcessPairs{"processpairs", false, "Process pairs: false = no, just singles, true = yes, process pairs"};

  Configurable<int> cfgPtBins{"ptbins", 18, "Number of pT bins, default 18"};
  Configurable<int> cfgZVtxBins{"zvtxbins", 28, "Number of z_vtx bins, default 28"};
  Configurable<int> cfgEtaBins{"etabins", 16, "Number of eta bins, default 16"};
  Configurable<float> cfgPtLow{"ptlow", 0.2f, "Lowest pT (GeV/c), default 0.2"};
  Configurable<float> cfgPtHigh{"pthigh", 2.0f, "Highest pT (GeV/c), default 2.0"};
  Configurable<float> cfgZVtxLow{"zvtxlow", -7.0f, "Lowest z_vtx distance (cm), default -7.0"};
  Configurable<float> cfgZVtxHigh{"zvtxhigh", 7.0f, "Highest z_vtx distance (cm), default 7.0"};
  Configurable<float> cfgEtaLow{"etalow", -0.8f, "Lowest eta value, default -0.8"};
  Configurable<float> cfgEtaHigh{"etahigh", 0.8f, "Highest eta value, default 0.8"};

  OutputObj<TList> fOutput{"DptDptCorrelationsData", OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext const&)
  {
    /* update the configurable values */
    ptbins = cfgPtBins.value;
    ptlow = cfgPtLow.value;
    ptup = cfgPtHigh.value;
    etabins = cfgEtaBins.value;
    etalow = cfgEtaLow.value;
    etaup = cfgEtaHigh.value;
    zvtxbins = cfgZVtxBins.value;
    zvtxlow = cfgZVtxLow.value;
    zvtxup = cfgZVtxHigh.value;
    tracktype = cfgTrackType.value;
    trackonecharge = cfgTrackOneCharge.value;
    tracktwocharge = cfgTrackTwoCharge.value;
    processpairs = cfgProcessPairs.value;

    /* still missing configuration */
    phibins = 72;
    /* TODO: shift of the azimuthal angle origin */
    philow = 0.0;
    phiup = TMath::TwoPi();

    /* create the output list which will own the task histograms */
    TList* fOutputList = new TList();
    fOutputList->SetOwner(true);
    fOutput.setObject(fOutputList);

    /* incorporate configuration parameters to the output */
    fOutputList->Add(new TParameter<Int_t>("NoBinsVertexZ", zvtxbins, 'f'));
    fOutputList->Add(new TParameter<Int_t>("NoBinsPt", ptbins, 'f'));
    fOutputList->Add(new TParameter<Int_t>("NoBinsEta", etabins, 'f'));
    fOutputList->Add(new TParameter<Int_t>("NoBinsPhi", phibins, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MinVertexZ", zvtxlow, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MaxVertexZ", zvtxup, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MinPt", ptlow, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MaxPt", ptup, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MinEta", etalow, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MaxEta", etaup, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MinPhi", philow, 'f'));
    fOutputList->Add(new TParameter<Double_t>("MaxPhi", phiup, 'f'));

    /* TODO: the shift in the origen of the azimuthal angle */

    /* create the histograms */
    Bool_t oldstatus = TH1::AddDirectoryStatus();
    TH1::AddDirectory(kFALSE);

    if (!processpairs) {
      fhN1_1_vsPt = new TH1F("n1_1_vsPt", "#LT n_{1} #GT;p_{t,1} (GeV/c);#LT n_{1} #GT", ptbins, ptlow, ptup);
      fhN1_1_vsZEtaPhiPt = new TH3F("n1_1_vsZ_vsEtaPhi_vsPt", "#LT n_{1} #GT;vtx_{z};#eta_{1}#times#varphi_{1};p_{t,1} (GeV/c)",
                                    zvtxbins, zvtxlow, zvtxup, etabins * phibins, 0.0, double(etabins * phibins), ptbins, ptlow, ptup);

      fhN1_2_vsPt = new TH1F("n1_2_vsPt", "#LT n_{1} #GT;p_{t,2} (GeV/c);#LT n_{1} #GT", ptbins, ptlow, ptup);
      fhN1_2_vsZEtaPhiPt = new TH3F("n1_2_vsZ_vsEtaPhi_vsPt", "#LT n_{2} #GT;vtx_{z};#eta_{2}#times#varphi_{2};p_{t,2} (GeV/c)",
                                    zvtxbins, zvtxlow, zvtxup, etabins * phibins, 0.0, double(etabins * phibins), ptbins, ptlow, ptup);

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

      fhN2_12_vsEtaPhi = new TH1F("n2_12_vsEtaPhi", "#LT n_{2} #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT n_{2} #GT",
                                  etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
      fhSum2PtPt_12_vsEtaPhi = new TH1F("sumPtPt_12_vsEtaPhi", "#LT #Sigma p_{t,1}p_{t,2} #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT #Sigma p_{t,1}p_{t,2} #GT (GeV)^{2}",
                                        etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
      fhSum2PtN_12_vsEtaPhi = new TH1F("sumPtN_12_vsEtaPhi", "#LT #Sigma p_{t,1}N #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT #Sigma p_{t,1}N #GT (GeV)",
                                       etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
      fhSum2NPt_12_vsEtaPhi = new TH1F("sumNPt_12_vsEtaPhi", "#LT N#Sigma p_{t,2} #GT;#eta_{1}#times#varphi_{1}#times#eta_{2}#times#varphi_{2};#LT N#Sigma p_{t,2} #GT (GeV)",
                                       etabins * phibins * etabins * phibins, 0., double(etabins * phibins * etabins * phibins));
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

      /* the statistical uncertainties will be estimated by the subsamples method so let's get rid of error tracking */
      fhN2_12_vsEtaPhi->Sumw2(false);
      fhSum2PtPt_12_vsEtaPhi->Sumw2(false);
      fhSum2PtN_12_vsEtaPhi->Sumw2(false);
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

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == DPTDPT_TRUE);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE) or (aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE));

  Partition<aod::FilteredTracks> Tracks1 = aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE;
  Partition<aod::FilteredTracks> Tracks2 = aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE;

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents, aod::AcceptedEvents>>::iterator const& collision,
               aod::FilteredTracks const& tracks)
  {
    if (!processpairs) {
      /* process single tracks */
      for (auto& track1 : Tracks1) {
        double corr = 1.0; /* TODO: track correction  weights */
        fhN1_1_vsPt->Fill(track1.pt(), corr);
        fhN1_1_vsZEtaPhiPt->Fill(collision.posZ(), GetEtaPhiIndex(track1.eta(), track1.phi()) + 0.5, track1.pt(), corr);
      }
      for (auto& track2 : Tracks2) {
        double corr = 1.0; /* TODO: track correction  weights */
        fhN1_2_vsPt->Fill(track2.pt(), corr);
        fhN1_2_vsZEtaPhiPt->Fill(collision.posZ(), GetEtaPhiIndex(track2.eta(), track2.phi()) + 0.5, track2.pt(), corr);
      }
    } else {
      /* process pairs of tracks */
      // this will be desireable
      //      for (auto& [track1, track2] : combinations(CombinationsFullIndexPolicy(Tracks1, Tracks2))) {
      for (auto& track1 : Tracks1) {
        for (auto& track2 : Tracks2) {
          /* checkiing the same track id condition */
          if (track1.index() == track2.index()) {
            LOGF(INFO, "Tracks with the same Id: %d", track1.index());
          } else {
          }
        }
      }
    }
  }
};

// Task for building <dpt,dpt> correlations
struct TracksAndEventClassificationQA {

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

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<DptDptCorrelationsFilterAnalysisTask>("DptDptCorrelationsFilterAnalysisTask"),
    adaptAnalysisTask<DptDptCorrelationsTask>("DptDptCorrelationsTask"),
    adaptAnalysisTask<TracksAndEventClassificationQA>("TracksAndEventClassificationQA"),
  };
}
