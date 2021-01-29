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
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTPC.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"add-qa", VariantType::Int, 0, {"Produce TPC PID QA histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct pidTPCTask {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collisions;
  Produces<aod::pidRespTPC> tpcpid;
  DetectorResponse resp;
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
      resp.LoadParamFromFile(fname.data(), signalname.value, DetectorResponse::kSignal);
      resp.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TPC";
      resp.LoadParam(DetectorResponse::kSignal, ccdb->getForTimeStamp<Parametrization>(path + "/" + signalname.value, timestamp.value));
      resp.LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
    }
  }

  void process(Coll const& collisions, Trks const& tracks)
  {
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Electron> resp_Electron = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Electron>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Muon> resp_Muon = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Muon>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Pion> resp_Pion = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Pion>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Kaon> resp_Kaon = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Kaon>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Proton> resp_Proton = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Proton>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Deuteron> resp_Deuteron = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Deuteron>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Triton> resp_Triton = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Triton>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Helium3> resp_Helium3 = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Helium3>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Alpha> resp_Alpha = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Alpha>();

    tpcpid.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tpcpid(resp_Electron.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Muon.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Pion.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Kaon.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Proton.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Deuteron.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Triton.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Helium3.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Alpha.GetExpectedSigma(resp, trk.collision(), trk),
             resp_Electron.GetSeparation(resp, trk.collision(), trk),
             resp_Muon.GetSeparation(resp, trk.collision(), trk),
             resp_Pion.GetSeparation(resp, trk.collision(), trk),
             resp_Kaon.GetSeparation(resp, trk.collision(), trk),
             resp_Proton.GetSeparation(resp, trk.collision(), trk),
             resp_Deuteron.GetSeparation(resp, trk.collision(), trk),
             resp_Triton.GetSeparation(resp, trk.collision(), trk),
             resp_Helium3.GetSeparation(resp, trk.collision(), trk),
             resp_Alpha.GetSeparation(resp, trk.collision(), trk));
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

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC> const& tracks)
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
  auto workflow = WorkflowSpec{adaptAnalysisTask<pidTPCTask>("pidTPC-task")};
  if (cfgc.options().get<int>("add-qa")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskQA>("pidTPCQA-task"));
  }
  return workflow;
}
