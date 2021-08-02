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
#include "AnalysisCore/MC.h"
#include "AnalysisConfigurableCuts.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include <TROOT.h>
#include <TDatabasePDG.h>
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
  ConfigParamSpec multest = {"wfcentmultestimator",
                             VariantType::String,
                             "NOCM",
                             {"Centrality/multiplicity estimator detector at workflow creation level. A this level the default is NOCM"}};
  ConfigParamSpec ismc = {"isMCPROD",
                          VariantType::Bool,
                          true,
                          {"Analysis on MC data at workflow creation level. A this level the default is true"}};
  workflowOptions.push_back(multest);
  workflowOptions.push_back(ismc);
}

#include "Framework/runDataProcessing.h"

namespace o2
{
namespace aod
{
/* we have to change from int to bool when bool columns work properly */
namespace dptdptcorrelations
{
DECLARE_SOA_COLUMN(EventAccepted, eventaccepted, uint8_t); //! If the collision/event has been accepted or not
DECLARE_SOA_COLUMN(EventCentMult, centmult, float);        //! The centrality/multiplicity pecentile
} // namespace dptdptcorrelations
DECLARE_SOA_TABLE(AcceptedEvents, "AOD", "ACCEPTEDEVENTS", //! Accepted reconstructed collisions/events filtered table
                  o2::soa::Index<>,
                  collision::BCId,
                  collision::PosZ,
                  dptdptcorrelations::EventAccepted,
                  dptdptcorrelations::EventCentMult);
using AcceptedEvent = AcceptedEvents::iterator;
DECLARE_SOA_TABLE(AcceptedTrueEvents, "AOD", "ACCTRUEEVENTS", //! Accepted generated collisions/events filtered table
                  o2::soa::Index<>,
                  collision::BCId,
                  mccollision::PosZ,
                  dptdptcorrelations::EventAccepted,
                  dptdptcorrelations::EventCentMult);
using AcceptedTrueEvent = AcceptedTrueEvents::iterator;
namespace dptdptcorrelations
{
DECLARE_SOA_INDEX_COLUMN(AcceptedEvent, event);                      //! Reconstructed collision/event
DECLARE_SOA_INDEX_COLUMN(AcceptedTrueEvent, mcevent);                //! Generated collision/event
DECLARE_SOA_COLUMN(TrackacceptedAsOne, trackacceptedasone, uint8_t); //! Track accepted as type one
DECLARE_SOA_COLUMN(TrackacceptedAsTwo, trackacceptedastwo, uint8_t); //! Track accepted as type two
DECLARE_SOA_COLUMN(Pt, pt, float);                                   //! The track transverse momentum
DECLARE_SOA_COLUMN(Eta, eta, float);                                 //! The track pseudorapidity
DECLARE_SOA_COLUMN(Phi, phi, float);                                 //! The track azimuthal angle
} // namespace dptdptcorrelations
DECLARE_SOA_TABLE(ScannedTracks, "AOD", "SCANNEDTRACKS", //! The reconstructed tracks filtered table
                  dptdptcorrelations::AcceptedEventId,
                  dptdptcorrelations::TrackacceptedAsOne,
                  dptdptcorrelations::TrackacceptedAsTwo,
                  dptdptcorrelations::Pt,
                  dptdptcorrelations::Eta,
                  dptdptcorrelations::Phi);
DECLARE_SOA_TABLE(ScannedTrueTracks, "AOD", "SCANTRUETRACKS", //! The generated particles filtered table
                  dptdptcorrelations::AcceptedTrueEventId,
                  dptdptcorrelations::TrackacceptedAsOne,
                  dptdptcorrelations::TrackacceptedAsTwo,
                  dptdptcorrelations::Pt,
                  dptdptcorrelations::Eta,
                  dptdptcorrelations::Phi);

using CollisionsEvSelCent = soa::Join<aod::Collisions, aod::EvSels, aod::Cents>;
using CollisionEvSelCent = soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator;
using CollisionsEvSel = soa::Join<aod::Collisions, aod::EvSels>;
using CollisionEvSel = soa::Join<aod::Collisions, aod::EvSels>::iterator;
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

/// \enum GeneratorType
/// \brief Which kid of generator data is the task addressing
enum GenType {
  kData = 0, ///< actual data, not generated
  kMC,       ///< Generator level and detector level
  kFastMC,   ///< Gererator level but stored dataset
  kOnTheFly, ///< On the fly generator level data
  knGenData  ///< number of different generator data types
};

/// \enum CentMultEstimatorType
/// \brief The detector used to estimate centrality/multiplicity
enum CentMultEstimatorType {
  kNOCM = 0,           ///< do not use centrality/multiplicity estimator
  kV0M,                ///< V0M centrality/multiplicity estimator
  kV0A,                ///< V0A centrality/multiplicity estimator
  kV0C,                ///< V0C centrality/multiplicity estimator
  kCL0,                ///< CL0 centrality/multiplicity estimator
  kCL1,                ///< CL1 centrality/multiplicity estimator
  knCentMultEstimators ///< number of centrality/mutiplicity estimator
};

namespace filteranalysistask
{
//============================================================================================
// The DptDptCorrelationsFilterAnalysisTask output objects
//============================================================================================
SystemType fSystem = kNoSystem;
GenType fDataType = kData;
CentMultEstimatorType fCentMultEstimator = kV0M;
TDatabasePDG* fPDG = nullptr;
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

TH1F* fhTrueCentMultB = nullptr;
TH1F* fhTrueCentMultA = nullptr;
TH1F* fhTrueVertexZB = nullptr;
TH1F* fhTrueVertexZA = nullptr;
TH1F* fhTruePtB = nullptr;
TH1F* fhTruePtA = nullptr;
TH1F* fhTruePtPosB = nullptr;
TH1F* fhTruePtPosA = nullptr;
TH1F* fhTruePtNegB = nullptr;
TH1F* fhTruePtNegA = nullptr;

TH1F* fhTrueEtaB = nullptr;
TH1F* fhTrueEtaA = nullptr;

TH1F* fhTruePhiB = nullptr;
TH1F* fhTruePhiA = nullptr;

TH2F* fhTrueEtaVsPhiB = nullptr;
TH2F* fhTrueEtaVsPhiA = nullptr;

TH2F* fhTruePtVsEtaB = nullptr;
TH2F* fhTruePtVsEtaA = nullptr;
} // namespace filteranalysistask

namespace filteranalysistaskqa
{
TH1F* fhTracksOne = nullptr;
TH1F* fhTracksTwo = nullptr;
TH1F* fhTracksOneAndTwo = nullptr;
TH1F* fhTracksNone = nullptr;
TH1F* fhTracksOneUnsel = nullptr;
TH1F* fhTracksTwoUnsel = nullptr;
TH1F* fhTracksOneAndTwoUnsel = nullptr;
TH1F* fhTracksNoneUnsel = nullptr;
TH1F* fhSelectedEvents = nullptr;
} // namespace filteranalysistaskqa

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

/// \brief System type according to configuration string
/// \param sysstr The system configuration string
/// \return The internal code for the passed system string
SystemType getSystemType(std::string const& sysstr)
{
  /* we have to figure out how extract the system type */
  if (sysstr.empty() or (sysstr == "PbPb")) {
    return kPbPb;
  } else if (sysstr == "pp") {
    return kpp;
  } else if (sysstr == "pPb") {
    return kpPb;
  } else if (sysstr == "Pbp") {
    return kPbp;
  } else if (sysstr == "pPb") {
    return kpPb;
  } else if (sysstr == "XeXe") {
    return kXeXe;
  } else {
    LOGF(fatal, "DptDptCorrelations::getSystemType(). Wrong system type: %d", sysstr.c_str());
  }
  return kPbPb;
}

/// \brief Type of data according to the configuration string
/// \param datastr The data type configuration string
/// \return Internal code for the passed kind of data string
GenType getGenType(std::string const& datastr)
{
  /* we have to figure out how extract the type of data*/
  if (datastr.empty() or (datastr == "data")) {
    return kData;
  } else if (datastr == "MC") {
    return kMC;
  } else if (datastr == "FastMC") {
    return kFastMC;
  } else if (datastr == "OnTheFlyMC") {
    return kOnTheFly;
  } else {
    LOGF(fatal, "DptDptCorrelations::getGenType(). Wrong type of dat: %d", datastr.c_str());
  }
  return kData;
}

template <typename CollisionObject>
bool IsEvtSelected(CollisionObject const& collision, float& centormult)
{
  using namespace filteranalysistask;

  bool trigsel = false;
  if (fDataType != kData) {
    trigsel = true;
  } else if (collision.alias()[kINT7]) {
    if (collision.sel7()) {
      trigsel = true;
    }
  }

  bool zvtxsel = false;
  /* TODO: vertex quality checks */
  if (zvtxlow < collision.posZ() and collision.posZ() < zvtxup) {
    zvtxsel = true;
  }

  bool centmultsel = false;
  switch (fCentMultEstimator) {
    case kV0M:
      if (collision.centV0M() < 100 and 0 < collision.centV0M()) {
        centormult = collision.centV0M();
        centmultsel = true;
      }
      break;
    default:
      break;
  }
  return trigsel and zvtxsel and centmultsel;
}

template <typename CollisionObject>
bool IsEvtSelectedNoCentMult(CollisionObject const& collision, float& centormult)
{
  using namespace filteranalysistask;

  bool trigsel = false;
  if (fDataType != kData) {
    trigsel = true;
  } else if (collision.alias()[kINT7]) {
    if (collision.sel7() or collision.sel8()) {
      trigsel = true;
    }
  }

  bool zvtxsel = false;
  /* TODO: vertex quality checks */
  if (zvtxlow < collision.posZ() and collision.posZ() < zvtxup) {
    zvtxsel = true;
  }

  bool centmultsel = false;
  switch (fCentMultEstimator) {
    case kNOCM:
      centormult = 50.0;
      centmultsel = true;
      break;
    default:
      break;
  }
  return trigsel and zvtxsel and centmultsel;
}

template <typename CollisionObject>
bool IsTrueEvtSelected(CollisionObject const& collision, float centormult)
{
  using namespace filteranalysistask;

  bool zvtxsel = false;
  /* TODO: vertex quality checks */
  if (zvtxlow < collision.posZ() and collision.posZ() < zvtxup) {
    zvtxsel = true;
  }

  bool centmultsel = false;
  if (centormult < 100 and 0 < centormult) {
    centmultsel = true;
  }

  return zvtxsel and centmultsel;
}

template <typename TrackObject>
bool matchTrackType(TrackObject const& track)
{
  using namespace filteranalysistask;

  switch (tracktype) {
    case 1:
      if (track.isGlobalTrack() != 0 || track.isGlobalTrackSDD() != 0) {
        return true;
      } else {
        return false;
      }
      break;
    case 3:        /* Run3 track */
      return true; // so far we accept every kind of Run3 tracks
      break;
    default:
      return false;
  }
}

template <typename TrackObject>
inline void AcceptTrack(TrackObject const& track, bool& asone, bool& astwo)
{
  using namespace filteranalysistask;

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

template <typename ParticleObject, typename ParticleListObject>
inline void AcceptTrueTrack(ParticleObject& particle, ParticleListObject& particles, bool& asone, bool& astwo)
{
  using namespace filteranalysistask;

  asone = false;
  astwo = false;

  float charge = (fPDG->GetParticle(particle.pdgCode())->Charge() / 3 >= 1) ? 1.0 : ((fPDG->GetParticle(particle.pdgCode())->Charge() / 3 <= -1) ? -1.0 : 0.0);

  /* TODO: matchTrackType will not work. We need at least is physical primary */
  if (MC::isPhysicalPrimary(particle)) {
    if (ptlow < particle.pt() and particle.pt() < ptup and etalow < particle.eta() and particle.eta() < etaup) {
      if (((charge > 0) and (trackonecharge > 0)) or ((charge < 0) and (trackonecharge < 0))) {
        asone = true;
      }
      if (((charge > 0) and (tracktwocharge > 0)) or ((charge < 0) and (tracktwocharge < 0))) {
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
  Configurable<std::string> cfgDataType{"datatype", "data", "Data type: data, MC, FastMC, OnTheFlyMC. Default data"};
  Configurable<std::string> cfgSystem{"syst", "PbPb", "System: pp, PbPb, Pbp, pPb, XeXe. Default PbPb"};

  Configurable<o2::analysis::DptDptBinningCuts> cfgBinning{"binning",
                                                           {28, -7.0, 7.0, 18, 0.2, 2.0, 16, -0.8, 0.8, 72, 0.5},
                                                           "triplets - nbins, min, max - for z_vtx, pT, eta and phi, binning plus bin fraction of phi origin shift"};

  OutputObj<TList> fOutput{"DptDptCorrelationsGlobalInfo", OutputObjHandlingPolicy::AnalysisObject};

  Produces<aod::AcceptedEvents> acceptedevents;
  Produces<aod::ScannedTracks> scannedtracks;
  Produces<aod::AcceptedTrueEvents> acceptedtrueevents;
  Produces<aod::ScannedTrueTracks> scannedtruetracks;

  template <typename TrackListObject, typename CollisionIndex>
  void filterTracks(TrackListObject const& ftracks, CollisionIndex colix)
  {
    using namespace filteranalysistask;

    int acceptedtracks = 0;

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
        acceptedtracks++;
      }
      scannedtracks(colix, (uint8_t)asone, (uint8_t)astwo, track.pt(), track.eta(), track.phi());
    }
    LOGF(info, "Accepted %d reconstructed tracks", acceptedtracks);
  }

  template <typename ParticleListObject, typename CollisionIndex>
  void filterTrueTracks(ParticleListObject const& particles, CollisionIndex colix)
  {
    using namespace filteranalysistask;

    for (auto& particle : particles) {
      float charge = 0.0;
      TParticlePDG* pdgparticle = fPDG->GetParticle(particle.pdgCode());
      if (pdgparticle != nullptr) {
        charge = (pdgparticle->Charge() / 3 >= 1) ? 1.0 : ((pdgparticle->Charge() / 3 <= -1) ? -1.0 : 0.0);
      }

      /* before particle selection */
      fhTruePtB->Fill(particle.pt());
      fhTrueEtaB->Fill(particle.eta());
      fhTruePhiB->Fill(particle.phi());
      fhTrueEtaVsPhiB->Fill(particle.phi(), particle.eta());
      fhTruePtVsEtaB->Fill(particle.eta(), particle.pt());
      if (charge > 0) {
        fhTruePtPosB->Fill(particle.pt());
      } else if (charge < 0) {
        fhTruePtNegB->Fill(particle.pt());
      }

      /* track selection */
      /* tricky because the boolean columns issue */
      bool asone = false;
      bool astwo = false;
      if (charge != 0) {
        AcceptTrueTrack(particle, particles, asone, astwo);
        if (asone or astwo) {
          /* the track has been accepted */
          fhTruePtA->Fill(particle.pt());
          fhTrueEtaA->Fill(particle.eta());
          fhTruePhiA->Fill(particle.phi());
          fhTrueEtaVsPhiA->Fill(particle.phi(), particle.eta());
          fhTruePtVsEtaA->Fill(particle.eta(), particle.pt());
          if (charge > 0) {
            fhTruePtPosA->Fill(particle.pt());
          } else {
            fhTruePtNegA->Fill(particle.pt());
          }
        }
      }
      scannedtruetracks(colix, (uint8_t)asone, (uint8_t)astwo, particle.pt(), particle.eta(), particle.phi());
    }
  }

  void init(InitContext const&)
  {
    using namespace filteranalysistask;

    LOGF(info, "FilterAnalysisTask::init()");

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
    } else if (cfgCentMultEstimator->compare("NOCM") == 0) {
      fCentMultEstimator = kNOCM;
    } else {
      LOGF(fatal, "Centrality/Multiplicity estimator %s not supported yet", cfgCentMultEstimator->c_str());
    }

    /* if the system type is not known at this time, we have to put the initalization somewhere else */
    fSystem = getSystemType(cfgSystem);
    fDataType = getGenType(cfgDataType);
    fPDG = TDatabasePDG::Instance();

    /* create the output list which will own the task histograms */
    TList* fOutputList = new TList();
    fOutputList->SetOwner(true);
    fOutput.setObject(fOutputList);

    /* incorporate configuration parameters to the output */
    fOutputList->Add(new TParameter<Int_t>("TrackType", cfgTrackType, 'f'));
    fOutputList->Add(new TParameter<Int_t>("TrackOneCharge", trackonecharge, 'f'));
    fOutputList->Add(new TParameter<Int_t>("TrackTwoCharge", tracktwocharge, 'f'));

    if ((fDataType == kData) or (fDataType == kMC)) {
      /* create the reconstructed data histograms */
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
      fhPtPosB = new TH1F("fHistPtPosB", "P_{T} distribution for reconstructed (#plus) before;P_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
      fhPtPosA = new TH1F("fHistPtPosA", "P_{T} distribution for reconstructed (#plus);P_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
      fhPtNegB = new TH1F("fHistPtNegB", "P_{T} distribution for reconstructed (#minus) before;P_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
      fhPtNegA = new TH1F("fHistPtNegA", "P_{T} distribution for reconstructed (#minus);P_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
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

    if (fDataType != kData) {
      /* create the true data histograms */
      if (fSystem > kPbp) {
        fhTrueCentMultB = new TH1F("TrueCentralityB", "Centrality before (truth); centrality (%)", 100, 0, 100);
        fhTrueCentMultA = new TH1F("TrueCentralityA", "Centrality (truth); centrality (%)", 100, 0, 100);
      } else {
        /* for pp, pPb and Pbp systems use multiplicity instead */
        fhTrueCentMultB = new TH1F("TrueMultiplicityB", "Multiplicity before (truth); multiplicity (%)", 100, 0, 100);
        fhTrueCentMultA = new TH1F("TrueMultiplicityA", "Multiplicity (truth); multiplicity (%)", 100, 0, 100);
      }

      fhTrueVertexZB = new TH1F("TrueVertexZB", "Vertex Z before (truth); z_{vtx}", 60, -15, 15);
      fhTrueVertexZA = new TH1F("TrueVertexZA", "Vertex Z (truth); z_{vtx}", zvtxbins, zvtxlow, zvtxup);

      fhTruePtB = new TH1F("fTrueHistPtB", "p_{T} distribution before (truth);p_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
      fhTruePtA = new TH1F("fTrueHistPtA", "p_{T} distribution (truth);p_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
      fhTruePtPosB = new TH1F("fTrueHistPtPosB", "P_{T} distribution (#plus) before (truth);P_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
      fhTruePtPosA = new TH1F("fTrueHistPtPosA", "P_{T} distribution (#plus) (truth);P_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
      fhTruePtNegB = new TH1F("fTrueHistPtNegB", "P_{T} distribution (#minus) before (truth);P_{T} (GeV/c);dN/dP_{T} (c/GeV)", 100, 0.0, 15.0);
      fhTruePtNegA = new TH1F("fTrueHistPtNegA", "P_{T} distribution (#minus) (truth);P_{T} (GeV/c);dN/dP_{T} (c/GeV)", ptbins, ptlow, ptup);
      fhTrueEtaB = new TH1F("fTrueHistEtaB", "#eta distribution before (truth);#eta;counts", 40, -2.0, 2.0);
      fhTrueEtaA = new TH1F("fTrueHistEtaA", "#eta distribution (truth);#eta;counts", etabins, etalow, etaup);
      fhTruePhiB = new TH1F("fTrueHistPhiB", "#phi distribution before (truth);#phi;counts", 360, 0.0, 2 * M_PI);
      fhTruePhiA = new TH1F("fTrueHistPhiA", "#phi distribution (truth);#phi;counts", 360, 0.0, 2 * M_PI);
      fhTrueEtaVsPhiB = new TH2F(TString::Format("CSTaskTrueEtaVsPhiB_%s", fTaskConfigurationString.c_str()), "#eta vs #phi before (truth);#phi;#eta", 360, 0.0, 2 * M_PI, 100, -2.0, 2.0);
      fhTrueEtaVsPhiA = new TH2F(TString::Format("CSTaskTrueEtaVsPhiA_%s", fTaskConfigurationString.c_str()), "#eta vs #phi (truth);#phi;#eta", 360, 0.0, 2 * M_PI, etabins, etalow, etaup);
      fhTruePtVsEtaB = new TH2F(TString::Format("fhTruePtVsEtaB_%s", fTaskConfigurationString.c_str()), "p_{T} vs #eta before (truth);#eta;p_{T} (GeV/c)", etabins, etalow, etaup, 100, 0.0, 15.0);
      fhTruePtVsEtaA = new TH2F(TString::Format("fhTruePtVsEtaA_%s", fTaskConfigurationString.c_str()), "p_{T} vs #eta (truth);#eta;p_{T} (GeV/c)", etabins, etalow, etaup, ptbins, ptlow, ptup);

      /* add the hstograms to the output list */
      fOutputList->Add(fhTrueCentMultB);
      fOutputList->Add(fhTrueCentMultA);
      fOutputList->Add(fhTrueVertexZB);
      fOutputList->Add(fhTrueVertexZA);
      fOutputList->Add(fhTruePtB);
      fOutputList->Add(fhTruePtA);
      fOutputList->Add(fhTruePtPosB);
      fOutputList->Add(fhTruePtPosA);
      fOutputList->Add(fhTruePtNegB);
      fOutputList->Add(fhTruePtNegA);
      fOutputList->Add(fhTrueEtaB);
      fOutputList->Add(fhTrueEtaA);
      fOutputList->Add(fhTruePhiB);
      fOutputList->Add(fhTruePhiA);
      fOutputList->Add(fhTrueEtaVsPhiB);
      fOutputList->Add(fhTrueEtaVsPhiA);
      fOutputList->Add(fhTruePtVsEtaB);
      fOutputList->Add(fhTruePtVsEtaA);
    }
  }

  void processWithCent(aod::CollisionEvSelCent const& collision, soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended, aod::TrackSelection> const& ftracks)
  {
    using namespace filteranalysistask;

    // LOGF(info, "FilterAnalysisTask::processWithCent(). New collision with %d tracks", ftracks.size());

    fhCentMultB->Fill(collision.centV0M());
    fhVertexZB->Fill(collision.posZ());
    bool acceptedevent = false;
    float centormult = -100.0;
    if (IsEvtSelected(collision, centormult)) {
      acceptedevent = true;
      fhCentMultA->Fill(collision.centV0M());
      fhVertexZA->Fill(collision.posZ());
      acceptedevents(collision.bcId(), collision.posZ(), (uint8_t)acceptedevent, centormult);

      filterTracks(ftracks, acceptedevents.lastIndex());
    } else {
      acceptedevents(collision.bcId(), collision.posZ(), (uint8_t)acceptedevent, centormult);
      for (auto& track : ftracks) {
        scannedtracks(acceptedevents.lastIndex(), (uint8_t) false, (uint8_t) false, track.pt(), track.eta(), track.phi());
      }
    }
  }
  PROCESS_SWITCH(DptDptCorrelationsFilterAnalysisTask, processWithCent, "Process reco with centrality", false);

  void processWithoutCent(aod::CollisionEvSel const& collision, soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended, aod::TrackSelection> const& ftracks)
  {
    using namespace filteranalysistask;

    LOGF(info, "FilterAnalysisTask::processWithoutCent(). New collision with collision id %d and with %d tracks", collision.bcId(), ftracks.size());

    /* the task does not have access to either centrality nor multiplicity 
       classes information, so it has to live without it.
       For the time being we assign a value of 50% */
    fhCentMultB->Fill(50.0);
    fhVertexZB->Fill(collision.posZ());
    bool acceptedevent = false;
    float centormult = -100.0;
    if (IsEvtSelectedNoCentMult(collision, centormult)) {
      acceptedevent = true;
      fhCentMultA->Fill(50.0);
      fhVertexZA->Fill(collision.posZ());
      acceptedevents(collision.bcId(), collision.posZ(), (uint8_t)acceptedevent, centormult);

      filterTracks(ftracks, acceptedevents.lastIndex());
    } else {
      acceptedevents(collision.bcId(), collision.posZ(), (uint8_t)acceptedevent, centormult);
      for (auto& track : ftracks) {
        scannedtracks(acceptedevents.lastIndex(), (uint8_t) false, (uint8_t) false, track.pt(), track.eta(), track.phi());
      }
    }
  }
  PROCESS_SWITCH(DptDptCorrelationsFilterAnalysisTask, processWithoutCent, "Process reco without centrality", false);

  void processWithCentMC(aod::McCollision const& mccollision,
                         soa::Join<aod::McCollisionLabels, aod::Collisions, aod::EvSels, aod::Cents> const& collisions,
                         aod::McParticles const& mcparticles)
  {
    using namespace filteranalysistask;

    // LOGF(info, "FilterAnalysisTask::processWithCentMC(). New generated collision %d reconstructed collisions and %d particles", collisions.size(), mcparticles.size());

    /* TODO: in here we have to decide what to do in the following cases
       - On the fly production -> clearly we will need a different process
       - reconstructed collisions without generated associated -> we need a different process or a different signature
       - multiplicity/centrality classes extracted from the reconstructed collision but then
       - generated collision without associated reconstructed collision: how to extract mutliplicity/centrality classes?
       - generated collision with several associated reconstructed collisions: from which to extract multiplicity/centrality classes?
    */
    if (collisions.size() > 1) {
      LOGF(error, "FilterAnalysisTask::processWithCentMC(). Generated collision with more than one reconstructed collisions. Processing only the first for centrality/multiplicity classes extraction");
    }

    for (auto& collision : collisions) {
      float cent = collision.centV0M();
      fhTrueCentMultB->Fill(cent);
      fhTrueVertexZB->Fill(mccollision.posZ());

      bool acceptedevent = false;
      if (IsTrueEvtSelected(mccollision, cent)) {
        acceptedevent = true;
        fhTrueCentMultA->Fill(cent);
        fhTrueVertexZA->Fill(mccollision.posZ());
        acceptedtrueevents(mccollision.bcId(), mccollision.posZ(), (uint8_t)acceptedevent, cent);

        filterTrueTracks(mcparticles, acceptedtrueevents.lastIndex());
      } else {
        acceptedtrueevents(mccollision.bcId(), mccollision.posZ(), (uint8_t)acceptedevent, cent);
        for (auto& particle : mcparticles) {
          scannedtruetracks(acceptedtrueevents.lastIndex(), (uint8_t) false, (uint8_t) false, particle.pt(), particle.eta(), particle.phi());
        }
      }
      break; /* TODO: only processing the first reconstructed collision for centrality/multiplicity class estimation */
    }
  }
  PROCESS_SWITCH(DptDptCorrelationsFilterAnalysisTask, processWithCentMC, "Process generated with centrality", false);

  void processWithoutCentMC(aod::McCollision const& mccollision,
                            aod::McParticles const& mcparticles)
  {
    using namespace filteranalysistask;

    // LOGF(info, "FilterAnalysisTask::processWithoutCentMC(). New generated collision with %d particles", mcparticles.size());

    /* the task does not have access to either centrality nor multiplicity 
       classes information, so it has to live without it.
       For the time being we assign a value of 50% */
    fhTrueCentMultB->Fill(50.0);
    fhTrueVertexZB->Fill(mccollision.posZ());

    bool acceptedevent = false;
    float centormult = 50.0;
    if (IsTrueEvtSelected(mccollision, centormult)) {
      acceptedevent = true;
      fhTrueCentMultA->Fill(centormult);
      fhTrueVertexZA->Fill(mccollision.posZ());
      acceptedtrueevents(mccollision.bcId(), mccollision.posZ(), (uint8_t)acceptedevent, centormult);

      filterTrueTracks(mcparticles, acceptedtrueevents.lastIndex());
    } else {
      acceptedtrueevents(mccollision.bcId(), mccollision.posZ(), (uint8_t)acceptedevent, centormult);
      for (auto& particle : mcparticles) {
        scannedtruetracks(acceptedtrueevents.lastIndex(), (uint8_t) false, (uint8_t) false, particle.pt(), particle.eta(), particle.phi());
      }
    }
  }
  PROCESS_SWITCH(DptDptCorrelationsFilterAnalysisTask, processWithoutCentMC, "Process generated without centrality", false);
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
    TProfile* fhN1_vsC[2];           //!<! weighted single particle distribution vs event centrality/multiplicity, track 1 and 2
    TProfile* fhSum1Pt_vsC[2];       //!<! accumulated sum of weighted \f$p_T\f$ vs event centrality/multiplicity, track 1 and 2
    TProfile* fhN1nw_vsC[2];         //!<! un-weighted single particle distribution vs event centrality/multiplicity, track 1 and 2
    TProfile* fhSum1Ptnw_vsC[2];     //!<! accumulated sum of un-weighted \f$p_T\f$ vs event centrality/multiplicity, track 1 and 2
    TProfile* fhN2_vsC[4];           //!<! weighted accumulated two particle distribution vs event centrality/multiplicity 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2PtPt_vsC[4];     //!<! weighted accumulated \f${p_T}_1 {p_T}_2\f$ distribution vs event centrality/multiplicity 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2DptDpt_vsC[4];   //!<! weighted accumulated \f$\sum ({p_T}_1- <{p_T}_1>) ({p_T}_2 - <{p_T}_2>) \f$ distribution vs event centrality/multiplicity 1-1,1-2,2-1,2-2, combinations
    TProfile* fhN2nw_vsC[4];         //!<! un-weighted accumulated two particle distribution vs event centrality/multiplicity 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2PtPtnw_vsC[4];   //!<! un-weighted accumulated \f${p_T}_1 {p_T}_2\f$ distribution vs event centrality/multiplicity 1-1,1-2,2-1,2-2, combinations
    TProfile* fhSum2DptDptnw_vsC[4]; //!<! un-weighted accumulated \f$\sum ({p_T}_1- <{p_T}_1>) ({p_T}_2 - <{p_T}_2>) \f$ distribution vs \f$\Delta\eta,\;\Delta\phi\f$ distribution vs event centrality/multiplicity 1-1,1-2,2-1,2-2, combinations

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
      LOGF(INFO, "Processing %d tracks of type %d in a collision with cent/mult %f ", passedtracks.size(), tix, cmul);

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
    template <typename TrackOneListObject, typename TrackTwoListObject>
    void processTrackPairs(TrackOneListObject const& trks1, TrackTwoListObject const& trks2, int pix, float cmul)
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

    template <typename TrackOneListObject, typename TrackTwoListObject>
    void processCollision(TrackOneListObject const& Tracks1, TrackTwoListObject const& Tracks2, float zvtx, float centmult)
    {
      using namespace correlationstask;

      if (not processpairs) {
        /* process single tracks */
        processSingles(Tracks1, 0, zvtx); /* track one */
        processSingles(Tracks2, 1, zvtx); /* track two */
      } else {
        /* process track magnitudes */
        /* TODO: the centrality should be chosen non detector dependent */
        processTracks(Tracks1, 0, centmult); /* track one */
        processTracks(Tracks2, 1, centmult); /* track one */
        /* process pair magnitudes */
        processTrackPairs(Tracks1, Tracks1, kOO, centmult);
        processTrackPairs(Tracks1, Tracks2, kOT, centmult);
        processTrackPairs(Tracks2, Tracks1, kTO, centmult);
        processTrackPairs(Tracks2, Tracks2, kTT, centmult);
      }
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
                                     TString::Format("#LT n_{1} #GT (weighted);Centrality/Multiplicity (%%);#LT n_{1} #GT").Data(),
                                     100, 0.0, 100.0);
          fhSum1Pt_vsC[i] = new TProfile(TString::Format("sumPt_%s_vsM", tname[i]),
                                         TString::Format("#LT #Sigma p_{t,%s} #GT (weighted);Centrality/Multiplicity (%%);#LT #Sigma p_{t,%s} #GT (GeV/c)", tname[i], tname[i]).Data(),
                                         100, 0.0, 100.0);
          fhN1nw_vsC[i] = new TProfile(TString::Format("n1Nw_%s_vsM", tname[i]).Data(),
                                       TString::Format("#LT n_{1} #GT;Centrality/Multiplicity (%%);#LT n_{1} #GT").Data(),
                                       100, 0.0, 100.0);
          fhSum1Ptnw_vsC[i] = new TProfile(TString::Format("sumPtNw_%s_vsM", tname[i]).Data(),
                                           TString::Format("#LT #Sigma p_{t,%s} #GT;Centrality/Multiplicity (%%);#LT #Sigma p_{t,%s} #GT (GeV/c)", tname[i], tname[i]).Data(), 100, 0.0, 100.0);
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

          fhN2_vsC[i] = new TProfile(TString::Format("n2_12_vsM_%s", pname), TString::Format("#LT n_{2} #GT (%s) (weighted);Centrality/Multiplicity (%%);#LT n_{2} #GT", pname), 100, 0.0, 100.0);
          fhSum2PtPt_vsC[i] = new TProfile(TString::Format("sumPtPt_12_vsM_%s", pname), TString::Format("#LT #Sigma p_{t,1}p_{t,2} #GT (%s) (weighted);Centrality/Multiplicity (%%);#LT #Sigma p_{t,1}p_{t,2} #GT (GeV^{2})", pname), 100, 0.0, 100.0);
          fhSum2DptDpt_vsC[i] = new TProfile(TString::Format("sumDptDpt_12_vsM_%s", pname), TString::Format("#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (%s) (weighted);Centrality/Multiplicity (%%);#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (GeV^{2})", pname), 100, 0.0, 100.0);
          fhN2nw_vsC[i] = new TProfile(TString::Format("n2Nw_12_vsM_%s", pname), TString::Format("#LT n_{2} #GT (%s);Centrality/Multiplicity (%%);#LT n_{2} #GT", pname), 100, 0.0, 100.0);
          fhSum2PtPtnw_vsC[i] = new TProfile(TString::Format("sumPtPtNw_12_vsM_%s", pname), TString::Format("#LT #Sigma p_{t,1}p_{t,2} #GT (%s);Centrality/Multiplicity (%%);#LT #Sigma p_{t,1}p_{t,2} #GT (GeV^{2})", pname), 100, 0.0, 100.0);
          fhSum2DptDptnw_vsC[i] = new TProfile(TString::Format("sumDptDptNw_12_vsM_%s", pname), TString::Format("#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (%s);Centrality/Multiplicity (%%);#LT #Sigma (p_{t,1} - #LT p_{t,1} #GT)(p_{t,2} - #LT p_{t,2} #GT) #GT (GeV^{2})", pname), 100, 0.0, 100.0);

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

  /// \brief Get the data collecting engine index corresponding to the passed collision
  template <typename FilteredCollision>
  int getDCEindex(FilteredCollision collision)
  {
    int ixDCE = -1;
    float cm = collision.centmult();
    for (int i = 0; i < ncmranges; ++i) {
      if (cm < fCentMultMax[i]) {
        ixDCE = i;
        break;
      }
    }
    return ixDCE;
  }

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == (uint8_t) true);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true) or (aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true));

  void processRecLevel(soa::Filtered<aod::AcceptedEvents>::iterator const& collision, soa::Filtered<aod::ScannedTracks>& tracks)
  {
    using namespace correlationstask;

    /* locate the data collecting engine for the collision centrality/multiplicity */
    int ixDCE = getDCEindex(collision);
    if (not(ixDCE < 0)) {
      Partition<o2::aod::ScannedTracks> TracksOne = aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true;
      Partition<o2::aod::ScannedTracks> TracksTwo = aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true;
      TracksOne.bindTable(tracks);
      TracksTwo.bindTable(tracks);

      LOGF(INFO, "Accepted BC id %d collision with cent/mult %f and %d total tracks. Assigned DCE: %d", collision.bcId(), collision.centmult(), tracks.size(), ixDCE);
      LOGF(INFO, "Accepted new collision with cent/mult %f and %d type one tracks and %d type two tracks. Assigned DCE: %d", collision.centmult(), TracksOne.size(), TracksTwo.size(), ixDCE);
      dataCE[ixDCE]->processCollision(TracksOne, TracksTwo, collision.posZ(), collision.centmult());
    }
  }
  PROCESS_SWITCH(DptDptCorrelationsTask, processRecLevel, "Process reco level correlations", false);

  void processGenLevel(soa::Filtered<aod::AcceptedTrueEvents>::iterator const& collision, soa::Filtered<aod::ScannedTrueTracks>& tracks)
  {
    using namespace correlationstask;

    /* locate the data collecting engine for the collision centrality/multiplicity */
    int ixDCE = getDCEindex(collision);
    if (not(ixDCE < 0)) {
      Partition<o2::aod::ScannedTrueTracks> TracksOne = aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true;
      Partition<o2::aod::ScannedTrueTracks> TracksTwo = aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true;
      TracksOne.bindTable(tracks);
      TracksTwo.bindTable(tracks);

      LOGF(INFO, "Accepted BC id %d generated collision with cent/mult %f and %d total tracks. Assigned DCE: %d", collision.bcId(), collision.centmult(), tracks.size(), ixDCE);
      LOGF(INFO, "Accepted new generated collision with cent/mult %f and %d type one tracks and %d type two tracks. Assigned DCE: %d", collision.centmult(), TracksOne.size(), TracksTwo.size(), ixDCE);
      dataCE[ixDCE]->processCollision(TracksOne, TracksTwo, collision.posZ(), collision.centmult());
    }
  }
  PROCESS_SWITCH(DptDptCorrelationsTask, processGenLevel, "Process generator level correlations", false);
};

// Checking the filtered tables
/* it seems we cannot use a base class task */
// struct TracksAndEventClassificationQABase {

void initQATask(InitContext const&, TList* outlst)
{
  using namespace filteranalysistaskqa;

  fhTracksOne = new TH1F("TracksOne", "Tracks as track one;number of tracks;events", 1500, 0.0, 1500.0);
  fhTracksTwo = new TH1F("TracksTwo", "Tracks as track two;number of tracks;events", 1500, 0.0, 1500.0);
  fhTracksOneAndTwo = new TH1F("TracksOneAndTwo", "Tracks as track one and as track two;number of tracks;events", 1500, 0.0, 1500.0);
  fhTracksNone = new TH1F("TracksNone", "Not selected tracks;number of tracks;events", 1500, 0.0, 1500.0);
  fhTracksOneUnsel = new TH1F("TracksOneUnsel", "Tracks as track one;number of tracks;events", 1500, 0.0, 1500.0);
  fhTracksTwoUnsel = new TH1F("TracksTwoUnsel", "Tracks as track two;number of tracks;events", 1500, 0.0, 1500.0);
  fhTracksOneAndTwoUnsel = new TH1F("TracksOneAndTwoUnsel", "Tracks as track one and as track two;number of tracks;events", 1500, 0.0, 1500.0);
  fhTracksNoneUnsel = new TH1F("TracksNoneUnsel", "Not selected tracks;number of tracks;events", 1500, 0.0, 1500.0);
  fhSelectedEvents = new TH1F("SelectedEvents", "Selected events;;events", 2, 0.0, 2.0);
  fhSelectedEvents->GetXaxis()->SetBinLabel(1, "Not selected events");
  fhSelectedEvents->GetXaxis()->SetBinLabel(2, "Selected events");

  outlst->Add(fhTracksOne);
  outlst->Add(fhTracksTwo);
  outlst->Add(fhTracksOneAndTwo);
  outlst->Add(fhTracksNone);
  outlst->Add(fhTracksOneUnsel);
  outlst->Add(fhTracksTwoUnsel);
  outlst->Add(fhTracksOneAndTwoUnsel);
  outlst->Add(fhTracksNoneUnsel);
  outlst->Add(fhSelectedEvents);
}

template <typename FilteredCollision, typename FilteredTracks>
void processQATask(FilteredCollision const& collision,
                   FilteredTracks const& tracks)
{
  using namespace filteranalysistaskqa;

  if (collision.eventaccepted() != (uint8_t) true) {
    fhSelectedEvents->Fill(0.5);
  } else {
    fhSelectedEvents->Fill(1.5);
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
    fhTracksOneUnsel->Fill(ntracks_one);
    fhTracksTwoUnsel->Fill(ntracks_two);
    fhTracksNoneUnsel->Fill(ntracks_none);
    fhTracksOneAndTwoUnsel->Fill(ntracks_one_and_two);
  } else {
    fhTracksOne->Fill(ntracks_one);
    fhTracksTwo->Fill(ntracks_two);
    fhTracksNone->Fill(ntracks_none);
    fhTracksOneAndTwo->Fill(ntracks_one_and_two);
  }
}
// };

/* it seems we cannot use a base class task */
// struct TracksAndEventClassificationQARec : TracksAndEventClassificationQABase {
struct TracksAndEventClassificationQARec {
  OutputObj<TList> fOutput{"FliterTaskRecoQA", OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext const& context)
  {
    TList* fOutputList = new TList();
    fOutputList->SetName("FilterTaskRecoQA");
    fOutputList->SetOwner(true);
    fOutput.setObject(fOutputList);

    initQATask(context, fOutputList);
  }

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == (uint8_t) true);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true) or (aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true));

  void process(soa::Filtered<aod::AcceptedEvents>::iterator const& collision, soa::Filtered<aod::ScannedTracks> const& tracks)
  {
    LOGF(info, "New filtered collision with BC id %d and with %d accepted tracks", collision.bcId(), tracks.size());
    processQATask(collision, tracks);
  }
};

/* it seems we cannot use a base class task */
//struct TracksAndEventClassificationQAGen : TracksAndEventClassificationQABase {
struct TracksAndEventClassificationQAGen {
  OutputObj<TList> fOutput{"FliterTaskGenQA", OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext const& context)
  {
    TList* fOutputList = new TList();
    fOutputList->SetName("FilterTaskGenQA");
    fOutputList->SetOwner(true);
    fOutput.setObject(fOutputList);

    initQATask(context, fOutputList);
  }

  Filter onlyacceptedevents = (aod::dptdptcorrelations::eventaccepted == (uint8_t) true);
  Filter onlyacceptedtracks = ((aod::dptdptcorrelations::trackacceptedasone == (uint8_t) true) or (aod::dptdptcorrelations::trackacceptedastwo == (uint8_t) true));

  void process(soa::Filtered<aod::AcceptedTrueEvents>::iterator const& collision, soa::Filtered<aod::ScannedTrueTracks> const& tracks)
  {
    LOGF(info, "New filtered generated collision with BC id %d and with %d accepted tracks", collision.bcId(), tracks.size());
    processQATask(collision, tracks);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  std::string multest = cfgc.options().get<std::string>("wfcentmultestimator");
  bool ismc = cfgc.options().get<bool>("isMCPROD");
  if (ismc) {
    if (multest == "NOCM") {
      /* no centrality/multiplicity classes available */
      WorkflowSpec workflow{
        adaptAnalysisTask<DptDptCorrelationsFilterAnalysisTask>(cfgc, SetDefaultProcesses{{{"processWithoutCent", true}, {"processWithoutCentMC", true}}}),
        adaptAnalysisTask<TracksAndEventClassificationQARec>(cfgc),
        adaptAnalysisTask<TracksAndEventClassificationQAGen>(cfgc),
        adaptAnalysisTask<DptDptCorrelationsTask>(cfgc, TaskName{"DptDptCorrelationsTaskRec"}, SetDefaultProcesses{{{"processRecLevel", true}}}),
        adaptAnalysisTask<DptDptCorrelationsTask>(cfgc, TaskName{"DptDptCorrelationsTaskGen"}, SetDefaultProcesses{{{"processGenLevel", true}}})};
      return workflow;
    } else {
      /* centrality/multiplicity classes available */
      WorkflowSpec workflow{
        adaptAnalysisTask<DptDptCorrelationsFilterAnalysisTask>(cfgc, SetDefaultProcesses{{{"processWithCent", true}, {"processWithCentMC", true}}}),
        adaptAnalysisTask<TracksAndEventClassificationQARec>(cfgc),
        adaptAnalysisTask<TracksAndEventClassificationQAGen>(cfgc),
        adaptAnalysisTask<DptDptCorrelationsTask>(cfgc, TaskName{"DptDptCorrelationsTaskRec"}, SetDefaultProcesses{{{"processRecLevel", true}}}),
        adaptAnalysisTask<DptDptCorrelationsTask>(cfgc, TaskName{"DptDptCorrelationsTaskGen"}, SetDefaultProcesses{{{"processGenLevel", true}}})};
      return workflow;
    }
  } else {
    if (multest == "NOCM") {
      /* no centrality/multiplicity classes available */
      WorkflowSpec workflow{
        adaptAnalysisTask<DptDptCorrelationsFilterAnalysisTask>(cfgc, SetDefaultProcesses{{{"processWithoutCent", true}}}),
        adaptAnalysisTask<TracksAndEventClassificationQARec>(cfgc),
        adaptAnalysisTask<DptDptCorrelationsTask>(cfgc, TaskName{"DptDptCorrelationsTaskRec"}, SetDefaultProcesses{{{"processRecLevel", true}}})};
      return workflow;
    } else {
      /* centrality/multiplicity classes available */
      WorkflowSpec workflow{
        adaptAnalysisTask<DptDptCorrelationsFilterAnalysisTask>(cfgc, SetDefaultProcesses{{{"processWithCent", true}}}),
        adaptAnalysisTask<TracksAndEventClassificationQARec>(cfgc),
        adaptAnalysisTask<DptDptCorrelationsTask>(cfgc, TaskName{"DptDptCorrelationsTaskRec"}, SetDefaultProcesses{{{"processRecLevel", true}}})};
      return workflow;
    }
  }
}
