// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   pidTOF.cxx
/// \author Nicolo' Jacazio
/// \brief  Task to produce PID tables for TOF split for each particle with only the Nsigma information.
///         Only the tables for the mass hypotheses requested are filled, the others are sent empty.
///

// O2 includes
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/RunningWorkflowInfo.h"
#include "ReconstructionDataFormats/Track.h"
#include <CCDB/BasicCCDBManager.h>
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTOF.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"add-qa", VariantType::Int, 0, {"Produce TOF PID QA histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct tofPid {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov>;
  using Coll = aod::Collisions;
  // Tables to produce
  Produces<o2::aod::pidTOFEl> tablePIDEl;
  Produces<o2::aod::pidTOFMu> tablePIDMu;
  Produces<o2::aod::pidTOFPi> tablePIDPi;
  Produces<o2::aod::pidTOFKa> tablePIDKa;
  Produces<o2::aod::pidTOFPr> tablePIDPr;
  Produces<o2::aod::pidTOFDe> tablePIDDe;
  Produces<o2::aod::pidTOFTr> tablePIDTr;
  Produces<o2::aod::pidTOFHe> tablePIDHe;
  Produces<o2::aod::pidTOFAl> tablePIDAl;
  // Detector response and input parameters
  DetectorResponse response;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> paramfile{"param-file", "", "Path to the parametrization object, if emtpy the parametrization is not taken from file"};
  Configurable<std::string> sigmaname{"param-sigma", "TOFReso", "Name of the parametrization for the expected sigma, used in both file and CCDB mode"};
  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
  Configurable<std::string> ccdbPath{"ccdbPath", "Analysis/PID/TOF", "Path of the TOF parametrization on the CCDB"};
  Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};
  // Configuration flags to include and exclude particle hypotheses
  Configurable<int> pidEl{"pid-el", -1, {"Produce PID information for the Electron mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidMu{"pid-mu", -1, {"Produce PID information for the Muon mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidPi{"pid-pi", -1, {"Produce PID information for the Pion mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidKa{"pid-ka", -1, {"Produce PID information for the Kaon mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidPr{"pid-pr", -1, {"Produce PID information for the Proton mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidDe{"pid-de", -1, {"Produce PID information for the Deuterons mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidTr{"pid-tr", -1, {"Produce PID information for the Triton mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidHe{"pid-he", -1, {"Produce PID information for the Helium3 mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidAl{"pid-al", -1, {"Produce PID information for the Alpha mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};

  void init(o2::framework::InitContext& initContext)
  {
    // Checking the tables are requested in the workflow and enabling them
    auto& workflows = initContext.services().get<RunningWorkflowInfo const>();
    for (DeviceSpec device : workflows.devices) {
      for (auto input : device.inputs) {
        auto enableFlag = [&input](const std::string particle, Configurable<int>& flag) {
          const std::string table = "pidTOF" + particle;
          if (input.matcher.binding == table) {
            if (flag < 0) {
              flag.value = 1;
              LOG(INFO) << "Auto-enabling table: " + table;
            } else if (flag > 0) {
              flag.value = 1;
              LOG(INFO) << "Table enabled: " + table;
            } else {
              LOG(INFO) << "Table disabled: " + table;
            }
          }
        };
        enableFlag("El", pidEl);
        enableFlag("Mu", pidMu);
        enableFlag("Pi", pidPi);
        enableFlag("Ka", pidKa);
        enableFlag("Pr", pidPr);
        enableFlag("De", pidDe);
        enableFlag("Tr", pidTr);
        enableFlag("He", pidHe);
        enableFlag("Al", pidAl);
      }
    }
    // Getting the parametrization parameters
    ccdb->setURL(url.value);
    ccdb->setTimestamp(timestamp.value);
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    // Not later than now objects
    ccdb->setCreatedNotAfter(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    //
    const std::vector<float> p = {0.008, 0.008, 0.002, 40.0};
    response.SetParameters(DetectorResponse::kSigma, p);
    const std::string fname = paramfile.value;
    if (!fname.empty()) { // Loading the parametrization from file
      LOG(INFO) << "Loading exp. sigma parametrization from file" << fname << ", using param: " << sigmaname.value;
      response.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse::kSigma);
    } else { // Loading it from CCDB
      std::string path = ccdbPath.value + "/" + sigmaname.value;
      LOG(INFO) << "Loading exp. sigma parametrization from CCDB, using path: " << path << " for timestamp " << timestamp.value;
      response.LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path, timestamp.value));
    }
  }

  template <o2::track::PID::ID pid>
  using ResponseImplementation = tof::ExpTimes<Coll::iterator, Trks::iterator, pid>;
  void process(Coll const& collisions, Trks const& tracks)
  {
    constexpr auto responseEl = ResponseImplementation<PID::Electron>();
    constexpr auto responseMu = ResponseImplementation<PID::Muon>();
    constexpr auto responsePi = ResponseImplementation<PID::Pion>();
    constexpr auto responseKa = ResponseImplementation<PID::Kaon>();
    constexpr auto responsePr = ResponseImplementation<PID::Proton>();
    constexpr auto responseDe = ResponseImplementation<PID::Deuteron>();
    constexpr auto responseTr = ResponseImplementation<PID::Triton>();
    constexpr auto responseHe = ResponseImplementation<PID::Helium3>();
    constexpr auto responseAl = ResponseImplementation<PID::Alpha>();

    // Check and fill enabled tables
    auto makeTable = [&tracks](const Configurable<int>& flag, auto& table, const DetectorResponse& response, const auto& responsePID) {
      if (flag.value == 1) {
        // Prepare memory for enabled tables
        table.reserve(tracks.size());
        for (auto const& trk : tracks) { // Loop on Tracks
          const float separation = responsePID.GetSeparation(response, trk.collision(), trk);
          if (separation <= o2::aod::pidtof_tiny::binned_min) {
            table(o2::aod::pidtof_tiny::lower_bin);
          } else if (separation >= o2::aod::pidtof_tiny::binned_max) {
            table(o2::aod::pidtof_tiny::upper_bin);
          } else if (separation >= 0) {
            table(separation / o2::aod::pidtof_tiny::bin_width + 0.5f);
          } else {
            table(separation / o2::aod::pidtof_tiny::bin_width - 0.5f);
          }
        }
      }
    };
    makeTable(pidEl, tablePIDEl, response, responseEl);
    makeTable(pidMu, tablePIDMu, response, responseMu);
    makeTable(pidPi, tablePIDPi, response, responsePi);
    makeTable(pidKa, tablePIDKa, response, responseKa);
    makeTable(pidPr, tablePIDPr, response, responsePr);
    makeTable(pidDe, tablePIDDe, response, responseDe);
    makeTable(pidTr, tablePIDTr, response, responseTr);
    makeTable(pidHe, tablePIDHe, response, responseHe);
    makeTable(pidAl, tablePIDAl, response, responseAl);
  }
};

struct tofPidQa {

  static constexpr int Np = 9;
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr std::string_view hexpected[Np] = {"expected/El", "expected/Mu", "expected/Pi",
                                                     "expected/Ka", "expected/Pr", "expected/De",
                                                     "expected/Tr", "expected/He", "expected/Al"};
  static constexpr std::string_view hexpected_diff[Np] = {"expected_diff/El", "expected_diff/Mu", "expected_diff/Pi",
                                                          "expected_diff/Ka", "expected_diff/Pr", "expected_diff/De",
                                                          "expected_diff/Tr", "expected_diff/He", "expected_diff/Al"};
  static constexpr std::string_view hnsigma[Np] = {"nsigma/El", "nsigma/Mu", "nsigma/Pi",
                                                   "nsigma/Ka", "nsigma/Pr", "nsigma/De",
                                                   "nsigma/Tr", "nsigma/He", "nsigma/Al"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> logAxis{"logAxis", 1, "Flag to use a log momentum axis"};
  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0.1f, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 5.f, "Maximum momentum in range"};
  Configurable<int> nBinsDelta{"nBinsDelta", 200, "Number of bins for the Delta"};
  Configurable<float> MinDelta{"MinDelta", -1000.f, "Minimum Delta in range"};
  Configurable<float> MaxDelta{"MaxDelta", 1000.f, "Maximum Delta in range"};
  Configurable<int> nBinsNSigma{"nBinsNSigma", 200, "Number of bins for the NSigma"};
  Configurable<float> MinNSigma{"MinNSigma", -10.f, "Minimum NSigma in range"};
  Configurable<float> MaxNSigma{"MaxNSigma", 10.f, "Maximum NSigma in range"};

  template <typename T>
  void makelogaxis(T h)
  {
    if (logAxis == 0) {
      return;
    }
    const int nbins = h->GetNbinsX();
    double binp[nbins + 1];
    double max = h->GetXaxis()->GetBinUpEdge(nbins);
    double min = h->GetXaxis()->GetBinLowEdge(1);
    if (min <= 0) {
      min = 0.00001;
    }
    double lmin = TMath::Log10(min);
    double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins);
    for (int i = 0; i < nbins; i++) {
      binp[i] = TMath::Exp(TMath::Log(10) * (lmin + i * ldelta));
    }
    binp[nbins] = max + 1;
    h->GetXaxis()->Set(nbins, binp);
  }

  template <uint8_t i>
  void addParticleHistos()
  {
    // NSigma
    histos.add(hnsigma[i].data(), Form(";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {nBinsNSigma, MinNSigma, MaxNSigma}});
    makelogaxis(histos.get<TH2>(HIST(hnsigma[i])));
  }

  void init(o2::framework::InitContext&)
  {
    // Event properties
    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", HistType::kTH1F, {{100, -20, 20}});
    histos.add("event/colltime", ";Collision time (ps);Entries", HistType::kTH1F, {{100, -2000, 2000}});
    histos.add("event/tofsignal", ";#it{p} (GeV/#it{c});TOF Signal", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {10000, 0, 2e6}});
    makelogaxis(histos.get<TH2>(HIST("event/tofsignal")));
    histos.add("event/eta", ";#it{#eta};Entries", HistType::kTH1F, {{100, -2, 2}});
    histos.add("event/length", ";Track length (cm);Entries", HistType::kTH1F, {{100, 0, 500}});
    histos.add("event/pt", ";#it{p}_{T} (GeV/#it{c});Entries", HistType::kTH1F, {{nBinsP, MinP, MaxP}});
    histos.add("event/p", ";#it{p} (GeV/#it{c});Entries", HistType::kTH1F, {{nBinsP, MinP, MaxP}});
    histos.add("event/ptreso", ";#it{p} (GeV/#it{c});Entries", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {100, 0, 0.1}});

    addParticleHistos<0>();
    addParticleHistos<1>();
    addParticleHistos<2>();
    addParticleHistos<3>();
    addParticleHistos<4>();
    addParticleHistos<5>();
    addParticleHistos<6>();
    addParticleHistos<7>();
    addParticleHistos<8>();
  }

  template <uint8_t i, typename T>
  void fillParticleHistos(const T& t, const float nsigma)
  {
    histos.fill(HIST(hnsigma[i]), t.p(), nsigma);
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov,
                                                          aod::pidTOFEl, aod::pidTOFMu, aod::pidTOFPi,
                                                          aod::pidTOFKa, aod::pidTOFPr, aod::pidTOFDe,
                                                          aod::pidTOFTr, aod::pidTOFHe, aod::pidTOFAl,
                                                          aod::TrackSelection> const& tracks)
  {
    const float collisionTime_ps = collision.collisionTime() * 1000.f;
    histos.fill(HIST("event/vertexz"), collision.posZ());
    histos.fill(HIST("event/colltime"), collisionTime_ps);

    for (auto t : tracks) {
      //
      if (!t.hasTOF()) { // Skipping tracks without TOF
        continue;
      }
      if (!t.isGlobalTrack()) {
        continue;
      }

      const float tof = t.tofSignal() - collisionTime_ps;

      //
      histos.fill(HIST("event/tofsignal"), t.p(), t.tofSignal());
      histos.fill(HIST("event/eta"), t.eta());
      histos.fill(HIST("event/length"), t.length());
      histos.fill(HIST("event/pt"), t.pt());
      histos.fill(HIST("event/ptreso"), t.p(), t.sigma1Pt() * t.pt() * t.pt());
      //
      fillParticleHistos<0>(t, t.tofNSigmaEl());
      fillParticleHistos<1>(t, t.tofNSigmaMu());
      fillParticleHistos<2>(t, t.tofNSigmaPi());
      fillParticleHistos<3>(t, t.tofNSigmaKa());
      fillParticleHistos<4>(t, t.tofNSigmaPr());
      fillParticleHistos<5>(t, t.tofNSigmaDe());
      fillParticleHistos<6>(t, t.tofNSigmaTr());
      fillParticleHistos<7>(t, t.tofNSigmaHe());
      fillParticleHistos<8>(t, t.tofNSigmaAl());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{adaptAnalysisTask<tofPid>(cfgc)};
  if (cfgc.options().get<int>("add-qa")) {
    workflow.push_back(adaptAnalysisTask<tofPidQa>(cfgc));
  }
  return workflow;
}
