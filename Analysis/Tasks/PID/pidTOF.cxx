// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// O2 includes
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/Track.h"
#include <CCDB/BasicCCDBManager.h>
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTOF.h"
#include "AnalysisDataModel/PID/TOFReso.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"use-fast", VariantType::Int, 0, {"Use fast implementation"}},
    {"add-qa", VariantType::Int, 0, {"Produce TOF PID QA histograms"}},
    {"add-beta", VariantType::Int, 1, {"Produce TOF Beta table"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct pidTOFTask {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov>;
  using Coll = aod::Collisions;
  Produces<aod::pidRespTOF> tablePID;
  DetectorResponse response;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> paramfile{"param-file", "", "Path to the parametrization object, if emtpy the parametrization is not taken from file"};
  Configurable<std::string> sigmaname{"param-sigma", "TOFReso", "Name of the parametrization for the expected sigma, used in both file and CCDB mode"};
  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
  Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};

  void init(o2::framework::InitContext&)
  {
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
      LOG(INFO) << "Loading parametrization from file" << fname << ", using param: " << sigmaname;
      response.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TOF";
      response.LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
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

    tablePID.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tablePID(responseEl.GetExpectedSigma(response, trk.collision(), trk),
               responseMu.GetExpectedSigma(response, trk.collision(), trk),
               responsePi.GetExpectedSigma(response, trk.collision(), trk),
               responseKa.GetExpectedSigma(response, trk.collision(), trk),
               responsePr.GetExpectedSigma(response, trk.collision(), trk),
               responseDe.GetExpectedSigma(response, trk.collision(), trk),
               responseTr.GetExpectedSigma(response, trk.collision(), trk),
               responseHe.GetExpectedSigma(response, trk.collision(), trk),
               responseAl.GetExpectedSigma(response, trk.collision(), trk),
               responseEl.GetSeparation(response, trk.collision(), trk),
               responseMu.GetSeparation(response, trk.collision(), trk),
               responsePi.GetSeparation(response, trk.collision(), trk),
               responseKa.GetSeparation(response, trk.collision(), trk),
               responsePr.GetSeparation(response, trk.collision(), trk),
               responseDe.GetSeparation(response, trk.collision(), trk),
               responseTr.GetSeparation(response, trk.collision(), trk),
               responseHe.GetSeparation(response, trk.collision(), trk),
               responseAl.GetSeparation(response, trk.collision(), trk));
    }
  }
};

struct pidTOFTaskBeta {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collision;
  Produces<aod::pidRespTOFbeta> tablePIDBeta;
  tof::Beta<Coll, Trks::iterator, PID::Electron> responseElectron;
  Configurable<float> expreso{"tof-expreso", 80, "Expected resolution for the computation of the expected beta"};

  void init(o2::framework::InitContext&)
  {
    responseElectron.mExpectedResolution = expreso.value;
  }

  void process(Coll const& collision, Trks const& tracks)
  {
    tablePIDBeta.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tablePIDBeta(responseElectron.GetBeta(collision, trk),
                   responseElectron.GetExpectedSigma(collision, trk),
                   responseElectron.GetExpectedSignal(collision, trk),
                   responseElectron.GetExpectedSigma(collision, trk),
                   responseElectron.GetSeparation(collision, trk));
    }
  }
};

struct pidTOFTaskFast {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov>;
  using Coll = aod::Collisions;
  Produces<aod::pidRespTOF> tablePID;
  DetectorResponse response;
  Parameters resoParameters;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> paramfile{"param-file", "", "Path to the parametrization object, if emtpy the parametrization is not taken from file"};
  Configurable<std::string> sigmaname{"param-sigma", "TOFReso", "Name of the parametrization for the expected sigma, used in both file and CCDB mode"};
  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
  Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};

  void init(o2::framework::InitContext&)
  {
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
      LOG(INFO) << "Loading parametrization from file" << fname << ", using param: " << sigmaname;
      response.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TOF";
      response.LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
    }
    resoParameters = response.GetParam(DetectorResponse::kSigma)->GetParameters();
  }

  template <o2::track::PID::ID id>
  float sigma(Trks::iterator track)
  {
    return o2::pid::tof::TOFResoParamTrack<id>(track.collision(), track, resoParameters);
  }
  template <o2::track::PID::ID id>
  float nsigma(Trks::iterator track)
  {
    return (track.tofSignal() - track.collision().collisionTime() * 1000.f - tof::ExpTimes<Coll::iterator, Trks::iterator, id>::GetExpectedSignal(track.collision(), track)) / sigma<id>(track);
  }
  void process(Coll const& collisions, Trks const& tracks)
  {
    tablePID.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tablePID(sigma<PID::Electron>(trk),
               sigma<PID::Muon>(trk),
               sigma<PID::Pion>(trk),
               sigma<PID::Kaon>(trk),
               sigma<PID::Proton>(trk),
               sigma<PID::Deuteron>(trk),
               sigma<PID::Triton>(trk),
               sigma<PID::Helium3>(trk),
               sigma<PID::Alpha>(trk),
               nsigma<PID::Electron>(trk),
               nsigma<PID::Muon>(trk),
               nsigma<PID::Pion>(trk),
               nsigma<PID::Kaon>(trk),
               nsigma<PID::Proton>(trk),
               nsigma<PID::Deuteron>(trk),
               nsigma<PID::Triton>(trk),
               nsigma<PID::Helium3>(trk),
               nsigma<PID::Alpha>(trk));
    }
  }
};

struct pidTOFTaskQA {

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
    // Exp signal
    histos.add(hexpected[i].data(), Form(";#it{p} (GeV/#it{c});t_{exp}(%s)", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 2e6}});
    makelogaxis(histos.get<TH2>(HIST(hexpected[i])));

    // T-Texp
    histos.add(hexpected_diff[i].data(), Form(";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp}(%s))", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {nBinsDelta, MinDelta, MaxDelta}});
    makelogaxis(histos.get<TH2>(HIST(hexpected_diff[i])));

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
    histos.add("event/tofbeta", ";#it{p} (GeV/#it{c});TOF #beta", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 2}});
    makelogaxis(histos.get<TH2>(HIST("event/tofbeta")));
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
  void fillParticleHistos(const T& t, const float tof, const float exp_diff, const float nsigma)
  {
    histos.fill(HIST(hexpected[i]), t.p(), tof - exp_diff);
    histos.fill(HIST(hexpected_diff[i]), t.p(), exp_diff);
    histos.fill(HIST(hnsigma[i]), t.p(), nsigma);
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::pidRespTOF, aod::pidRespTOFbeta, aod::TrackSelection> const& tracks)
  {
    const float collisionTime_ps = collision.collisionTime() * 1000.f;
    histos.fill(HIST("event/vertexz"), collision.posZ());
    histos.fill(HIST("event/colltime"), collisionTime_ps);

    for (auto t : tracks) {
      //
      if (t.tofSignal() < 0) { // Skipping tracks without TOF
        continue;
      }
      if (!t.isGlobalTrack()) {
        continue;
      }

      const float tof = t.tofSignal() - collisionTime_ps;

      //
      histos.fill(HIST("event/tofsignal"), t.p(), t.tofSignal());
      histos.fill(HIST("event/tofbeta"), t.p(), t.beta());
      histos.fill(HIST("event/eta"), t.eta());
      histos.fill(HIST("event/length"), t.length());
      histos.fill(HIST("event/pt"), t.pt());
      histos.fill(HIST("event/ptreso"), t.p(), t.sigma1Pt() * t.pt() * t.pt());
      //
      fillParticleHistos<0>(t, tof, t.tofExpSignalDiffEl(), t.tofNSigmaEl());
      fillParticleHistos<1>(t, tof, t.tofExpSignalDiffMu(), t.tofNSigmaMu());
      fillParticleHistos<2>(t, tof, t.tofExpSignalDiffPi(), t.tofNSigmaPi());
      fillParticleHistos<3>(t, tof, t.tofExpSignalDiffKa(), t.tofNSigmaKa());
      fillParticleHistos<4>(t, tof, t.tofExpSignalDiffPr(), t.tofNSigmaPr());
      fillParticleHistos<5>(t, tof, t.tofExpSignalDiffDe(), t.tofNSigmaDe());
      fillParticleHistos<6>(t, tof, t.tofExpSignalDiffTr(), t.tofNSigmaTr());
      fillParticleHistos<7>(t, tof, t.tofExpSignalDiffHe(), t.tofNSigmaHe());
      fillParticleHistos<8>(t, tof, t.tofExpSignalDiffAl(), t.tofNSigmaAl());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{};
  if (cfgc.options().get<int>("use-fast")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskFast>(cfgc, TaskName{"pidTOF-task"}));
  } else {
    workflow.push_back(adaptAnalysisTask<pidTOFTask>(cfgc, TaskName{"pidTOF-task"}));
  }
  const int add_beta = cfgc.options().get<int>("add-beta");
  const int add_qa = cfgc.options().get<int>("add-qa");
  if (add_beta || add_qa) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskBeta>(cfgc, TaskName{"pidTOFBeta-task"}));
  }
  if (add_qa) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA>(cfgc, TaskName{"pidTOFQA-task"}));
  }
  return workflow;
}
