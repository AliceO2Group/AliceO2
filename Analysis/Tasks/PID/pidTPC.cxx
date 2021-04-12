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
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/Track.h"
#include <CCDB/BasicCCDBManager.h>
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTPC.h"
#include "AnalysisDataModel/PID/TPCReso.h"
#include "AnalysisDataModel/PID/BetheBloch.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

// Table with the full information for all particles
namespace o2::aod
{
DECLARE_SOA_TABLE(pidRespTPC, "AOD", "pidRespTPC",
                  // Expected signals
                  pidtpc::TPCExpSignalDiffEl<pidtpc::TPCNSigmaEl, pidtpc::TPCExpSigmaEl>,
                  pidtpc::TPCExpSignalDiffMu<pidtpc::TPCNSigmaMu, pidtpc::TPCExpSigmaMu>,
                  pidtpc::TPCExpSignalDiffPi<pidtpc::TPCNSigmaPi, pidtpc::TPCExpSigmaPi>,
                  pidtpc::TPCExpSignalDiffKa<pidtpc::TPCNSigmaKa, pidtpc::TPCExpSigmaKa>,
                  pidtpc::TPCExpSignalDiffPr<pidtpc::TPCNSigmaPr, pidtpc::TPCExpSigmaPr>,
                  pidtpc::TPCExpSignalDiffDe<pidtpc::TPCNSigmaDe, pidtpc::TPCExpSigmaDe>,
                  pidtpc::TPCExpSignalDiffTr<pidtpc::TPCNSigmaTr, pidtpc::TPCExpSigmaTr>,
                  pidtpc::TPCExpSignalDiffHe<pidtpc::TPCNSigmaHe, pidtpc::TPCExpSigmaHe>,
                  pidtpc::TPCExpSignalDiffAl<pidtpc::TPCNSigmaAl, pidtpc::TPCExpSigmaAl>,
                  // Expected sigma
                  pidtpc::TPCExpSigmaEl, pidtpc::TPCExpSigmaMu, pidtpc::TPCExpSigmaPi,
                  pidtpc::TPCExpSigmaKa, pidtpc::TPCExpSigmaPr, pidtpc::TPCExpSigmaDe,
                  pidtpc::TPCExpSigmaTr, pidtpc::TPCExpSigmaHe, pidtpc::TPCExpSigmaAl,
                  // NSigma
                  pidtpc::TPCNSigmaEl, pidtpc::TPCNSigmaMu, pidtpc::TPCNSigmaPi,
                  pidtpc::TPCNSigmaKa, pidtpc::TPCNSigmaPr, pidtpc::TPCNSigmaDe,
                  pidtpc::TPCNSigmaTr, pidtpc::TPCNSigmaHe, pidtpc::TPCNSigmaAl);
}

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"use-fast", VariantType::Int, 0, {"Use fast implementation"}},
    {"add-qa", VariantType::Int, 0, {"Produce TPC PID QA histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct pidTPCTask {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collisions;
  Produces<aod::pidRespTPC> tablePID;
  DetectorResponse response;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> paramfile{"param-file", "", "Path to the parametrization object, if emtpy the parametrization is not taken from file"};
  Configurable<std::string> signalname{"param-signal", "BetheBloch", "Name of the parametrization for the expected signal, used in both file and CCDB mode"};
  Configurable<std::string> sigmaname{"param-sigma", "TPCReso", "Name of the parametrization for the expected sigma, used in both file and CCDB mode"};
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
    const std::string fname = paramfile.value;
    if (!fname.empty()) { // Loading the parametrization from file
      response.LoadParamFromFile(fname.data(), signalname.value, DetectorResponse::kSignal);
      response.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TPC";
      response.LoadParam(DetectorResponse::kSignal, ccdb->getForTimeStamp<Parametrization>(path + "/" + signalname.value, timestamp.value));
      response.LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
    }
  }

  template <o2::track::PID::ID pid>
  using ResponseImplementation = o2::pid::tpc::ELoss<Coll::iterator, Trks::iterator, pid>;
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

struct pidTPCTaskFast {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collisions;
  Produces<aod::pidRespTPC> tablePID;
  DetectorResponse response;
  Parameters expParameters;
  Parameters resoParameters;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> paramfile{"param-file", "", "Path to the parametrization object, if emtpy the parametrization is not taken from file"};
  Configurable<std::string> signalname{"param-signal", "BetheBloch", "Name of the parametrization for the expected signal, used in both file and CCDB mode"};
  Configurable<std::string> sigmaname{"param-sigma", "TPCReso", "Name of the parametrization for the expected sigma, used in both file and CCDB mode"};
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
    const std::string fname = paramfile.value;
    if (!fname.empty()) { // Loading the parametrization from file
      response.LoadParamFromFile(fname.data(), signalname.value, DetectorResponse::kSignal);
      response.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TPC";
      response.LoadParam(DetectorResponse::kSignal, ccdb->getForTimeStamp<Parametrization>(path + "/" + signalname.value, timestamp.value));
      response.LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
    }
    expParameters = response.GetParam(DetectorResponse::kSignal)->GetParameters();
    resoParameters = response.GetParam(DetectorResponse::kSigma)->GetParameters();
  }

  template <o2::track::PID::ID id>
  float exp(Trks::iterator track)
  {
    return o2::pid::tpc::BetheBlochParamTrack<id>(track, expParameters);
  }
  template <o2::track::PID::ID id>
  float sigma(Trks::iterator track)
  {
    return o2::pid::tpc::TPCResoParamTrack<id>(track, resoParameters);
  }
  template <o2::track::PID::ID id>
  float nsigma(Trks::iterator track)
  {
    return (track.tpcSignal() - exp<id>(track)) / sigma<id>(track);
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

struct pidTPCTaskQA {
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

  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 20, "Maximum momentum in range"};

  template <typename T>
  void makelogaxis(T h)
  {
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
    histos.add(hexpected[i].data(), Form(";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_(%s)", pT[i]), kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 1000}});
    makelogaxis(histos.get<TH2>(HIST(hexpected[i])));

    // Signal - Expected signal
    histos.add(hexpected_diff[i].data(), Form(";#it{p} (GeV/#it{c});;d#it{E}/d#it{x} - d#it{E}/d#it{x}(%s)", pT[i]), kTH2F, {{nBinsP, MinP, MaxP}, {1000, -500, 500}});
    makelogaxis(histos.get<TH2>(HIST(hexpected_diff[i])));

    // NSigma
    histos.add(hnsigma[i].data(), Form(";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(%s)", pT[i]), kTH2F, {{nBinsP, MinP, MaxP}, {200, -10, 10}});
    makelogaxis(histos.get<TH2>(HIST(hnsigma[i])));
  }

  void init(o2::framework::InitContext&)
  {
    // Event properties
    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", kTH1F, {{100, -20, 20}});
    histos.add("event/tpcsignal", ";#it{p} (GeV/#it{c});TPC Signal", kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 1000}});
    makelogaxis(histos.get<TH2>(HIST("event/tpcsignal")));

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
  void fillParticleHistos(const T& t, const float mom, const float exp_diff, const float nsigma)
  {
    histos.fill(HIST(hexpected[i]), mom, t.tpcSignal() - exp_diff);
    histos.fill(HIST(hexpected_diff[i]), mom, exp_diff);
    histos.fill(HIST(hnsigma[i]), t.p(), nsigma);
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::TrackSelection> const& tracks)
  {
    histos.fill(HIST("event/vertexz"), collision.posZ());

    for (auto t : tracks) {
      // const float mom = t.p();
      const float mom = t.tpcInnerParam();
      histos.fill(HIST("event/tpcsignal"), mom, t.tpcSignal());
      //
      fillParticleHistos<0>(t, mom, t.tpcExpSignalDiffEl(), t.tpcNSigmaEl());
      fillParticleHistos<1>(t, mom, t.tpcExpSignalDiffMu(), t.tpcNSigmaMu());
      fillParticleHistos<2>(t, mom, t.tpcExpSignalDiffPi(), t.tpcNSigmaPi());
      fillParticleHistos<3>(t, mom, t.tpcExpSignalDiffKa(), t.tpcNSigmaKa());
      fillParticleHistos<4>(t, mom, t.tpcExpSignalDiffPr(), t.tpcNSigmaPr());
      fillParticleHistos<5>(t, mom, t.tpcExpSignalDiffDe(), t.tpcNSigmaDe());
      fillParticleHistos<6>(t, mom, t.tpcExpSignalDiffTr(), t.tpcNSigmaTr());
      fillParticleHistos<7>(t, mom, t.tpcExpSignalDiffHe(), t.tpcNSigmaHe());
      fillParticleHistos<8>(t, mom, t.tpcExpSignalDiffAl(), t.tpcNSigmaAl());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{};
  if (cfgc.options().get<int>("use-fast")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskFast>(cfgc, TaskName{"pidTPC-task"}));
  } else {
    workflow.push_back(adaptAnalysisTask<pidTPCTask>(cfgc, TaskName{"pidTPC-task"}));
  }
  if (cfgc.options().get<int>("add-qa")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskQA>(cfgc, TaskName{"pidTPCQA-task"}));
  }
  return workflow;
}
