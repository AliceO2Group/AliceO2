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
using CollisionEvSelCent = soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator;
using TrackData = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended, aod::TrackSelection>::iterator;

/* we have to change from int to bool when bool columns work properly */
namespace dptdptcorrelations
{
DECLARE_SOA_COLUMN(EventAccepted, eventaccepted, uint8_t);
DECLARE_SOA_COLUMN(TrackacceptedAsOne, trackacceptedasone, uint8_t);
DECLARE_SOA_COLUMN(TrackacceptedAsTwo, trackacceptedastwo, uint8_t);
} // namespace dptdptcorrelations
DECLARE_SOA_TABLE(AcceptedEvents, "AOD", "ACCEPTEDEVENTS", dptdptcorrelations::EventAccepted);
DECLARE_SOA_TABLE(ScannedTracks, "AOD", "SCANNEDTRACKS", dptdptcorrelations::TrackacceptedAsOne, dptdptcorrelations::TrackacceptedAsTwo);
} // namespace aod
} // namespace o2

namespace dptdptcorrelations
{
/* all this bins have to be make configurable */
int ptbins = 18;
float ptlow = 0.2, ptup = 2.0;
int etabins = 16;
float etalow = -0.8, etaup = 0.8;
int zvtxbins = 40;
float zvtxlow = -10.0, zvtxup = 10.0;
int phibins = 72;
std::string fTaskConfigurationString = "PendingToConfigure";

Configurable<int> cfgTrackType{"trktype", 1, "Type of selected tracks: 0 = no selection, 1 = global tracks loose DCA, 2 = global SDD tracks"};
Configurable<int> cfgTrackOneCharge("trk1charge", 1, "Trakc one charge: 1 = positive, -1 = negative");
Configurable<int> cfgTrackTwoCharge("trk2charge", -1, "Trakc two charge: 1 = positive, -1 = negative");
Configurable<bool> cfgProcessPairs("processpairs", true, "Process pairs: false = no, just singles, true = yes, process pairs");

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

/* probably this will not work in the context of distributed processes */
/* we need to revisit it!!!!                                           */
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
  switch (cfgTrackType) {
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

std::tuple<uint8_t, uint8_t> AcceptTrack(aod::TrackData const& track)
{

  uint8_t asone = DPTDPT_FALSE;
  uint8_t astwo = DPTDPT_FALSE;

  /* TODO: incorporate a mask in the scanned tracks table for the rejecting track reason */
  if (matchTrackType(track)) {
    if (ptlow < track.pt() and track.pt() < ptup and etalow < track.eta() and track.eta() < etaup) {
      if (((track.charge() > 0) and (cfgTrackOneCharge > 0)) or ((track.charge() < 0) and (cfgTrackOneCharge < 0))) {
        asone = DPTDPT_TRUE;
      } else if (((track.charge() > 0) and (cfgTrackTwoCharge > 0)) or ((track.charge() < 0) and (cfgTrackTwoCharge < 0))) {
        astwo = DPTDPT_TRUE;
      }
    }
  }
  return std::make_tuple(asone, astwo);
}
} /* end namespace dptdptcorrelations */

// Task for <dpt,dpt> correlations analysis
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that

using namespace dptdptcorrelations;

struct DptDptCorrelationsFilterAnalysisTask {

  OutputObj<TDirectory> fOutput{"DptDptCorrelationsGlobalInfo", OutputObjHandlingPolicy::AnalysisObject};

  OutputObj<TH1F> fOutCentMultB{"CentralityB", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutCentMultA{"CentralityA", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutVertexZB{"VertexZB", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutVertexZA{"VertexZA", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPtB{"fHistPtB", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPtA{"fHistPtA", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPtPosB{"fHistPtPosB", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPtPosA{"fHistPtPosA", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPtNegB{"fHistPtNegB", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPtNegA{"fHistPtNegA", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutEtaB{"fHistEtaB", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutEtaA{"fHistEtaA", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPhiB{"fHistPhiB", OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<TH1F> fOutPhiA{"fHistPhiA", OutputObjHandlingPolicy::AnalysisObject};
  //  no more histograms for the time being
  //  OutputObj<TH2F> fOutEtaVsPhiB{"CSTaskEtaVsPhiB", OutputObjHandlingPolicy::AnalysisObject};
  //  OutputObj<TH2F> fOutEtaVsPhiA{"CSTaskEtaVsPhiA", OutputObjHandlingPolicy::AnalysisObject};
  //  OutputObj<TH2F> fOutPtVsEtaB{"fhPtVsEtaB", OutputObjHandlingPolicy::AnalysisObject};
  //  OutputObj<TH2F> fOutPtVsEtaA{"fhPtVsEtaA", OutputObjHandlingPolicy::AnalysisObject};

  Produces<aod::AcceptedEvents> acceptedevents;
  Produces<aod::ScannedTracks> scannedtracks;

  void init(InitContext const&)
  {
    /* if the system type is not known at this time, we have to put the initalization somwhere else */
    fSystem = getSystemType();

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
    //  no more histograms for the time being
    //    fhEtaVsPhiB = new TH2F(TString::Format("CSTaskEtaVsPhiB_%s",fTaskConfigurationString.c_str()),"#eta vs #phi before;#phi;#eta", 360, 0.0, 2*M_PI, 100, -2.0, 2.0);
    //    fhEtaVsPhiA = new TH2F(TString::Format("CSTaskEtaVsPhiA_%s",fTaskConfigurationString.c_str()),"#eta vs #phi;#phi;#eta", 360, 0.0, 2*M_PI, etabins, etalow, etaup);
    //    fhPtVsEtaB = new TH2F(TString::Format("fhPtVsEtaB_%s",fTaskConfigurationString.c_str()),"p_{T} vs #eta before;#eta;p_{T} (GeV/c)",etabins,etalow,etaup,100,0.0,15.0);
    //    fhPtVsEtaA = new TH2F(TString::Format("fhPtVsEtaA_%s",fTaskConfigurationString.c_str()),"p_{T} vs #eta;#eta;p_{T} (GeV/c)",etabins,etalow,etaup,ptbins,ptlow,ptup);

    fOutCentMultB.setObject(fhCentMultB);
    fOutCentMultA.setObject(fhCentMultA);
    fOutVertexZB.setObject(fhVertexZB);
    fOutVertexZA.setObject(fhVertexZA);
    fOutPtB.setObject(fhPtB);
    fOutPtA.setObject(fhPtA);
    fOutPtPosB.setObject(fhPtPosB);
    fOutPtPosA.setObject(fhPtPosA);
    fOutPtNegB.setObject(fhPtNegB);
    fOutPtNegA.setObject(fhPtNegA);
    fOutEtaB.setObject(fhEtaB);
    fOutEtaA.setObject(fhEtaA);
    fOutPhiB.setObject(fhPhiB);
    fOutPhiA.setObject(fhPhiA);
    //  no more histograms for the time being
    //    fOutEtaVsPhiB.setObject(fhEtaVsPhiB);
    //    fOutEtaVsPhiA.setObject(fhEtaVsPhiA);
    //    fOutPtVsEtaB.setObject(fhPtVsEtaB);
    //    fOutPtVsEtaA.setObject(fhPtVsEtaA);
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
        //  no more histograms for the time being
        //        fhEtaVsPhiB->Fill(track.phi(),track.eta());
        //        fhPtVsEtaB->Fill(track.eta(),track.pt());
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
          //  no more histograms for the time being
          //          fhEtaVsPhiA->Fill(track.phi(),track.eta());
          //          fhPtVsEtaA->Fill(track.eta(),track.pt());
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

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == DPTDPT_TRUE);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE) or (aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE));

  using FilteredTracks = soa::Filtered<soa::Join<aod::Tracks, aod::ScannedTracks>>;
  Partition<FilteredTracks> Tracks1 = aod::dptdptcorrelations::trackacceptedasone == DPTDPT_TRUE;
  Partition<FilteredTracks> Tracks2 = aod::dptdptcorrelations::trackacceptedastwo == DPTDPT_TRUE;

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents, aod::AcceptedEvents>>::iterator const& collision,
               FilteredTracks const& tracks)
  {
    if (cfgProcessPairs) {
      /* process pairs of tracks */
      // this will be desireable
      //      for (auto& [track1, track2] : combinations(CombinationsFullIndexPolicy(Tracks1, Tracks2))) {
      for (auto& track1 : Tracks1) {
        for (auto& track2 : Tracks2) {
          /* checkiing the same track id condition */
          if (track1.index() == track2.index()) {
            LOGF(INFO, "Tracks with the same Id: %d", track1.index());
          }
          /* now it is just to collect the two particle information */
        }
      }
    } else {
      /* process single tracks */
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
  OutputObj<TH1F> fSelectedEvents{TH1F("SelectedEvents", "Selected events;number of tracks;events", 2, 0.0, 2.0)};

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
