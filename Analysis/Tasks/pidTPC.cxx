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
    {"add-qa", VariantType::Int, 0, {"Produce TOF PID QA histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct pidTPCTask {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collision;
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

  void process(Coll const& collision, Trks const& tracks)
  {
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Electron> resp_Electron = tpc::ELoss<Coll, Trks::iterator, PID::Electron>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Muon> resp_Muon = tpc::ELoss<Coll, Trks::iterator, PID::Muon>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Pion> resp_Pion = tpc::ELoss<Coll, Trks::iterator, PID::Pion>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Kaon> resp_Kaon = tpc::ELoss<Coll, Trks::iterator, PID::Kaon>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Proton> resp_Proton = tpc::ELoss<Coll, Trks::iterator, PID::Proton>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Deuteron> resp_Deuteron = tpc::ELoss<Coll, Trks::iterator, PID::Deuteron>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Triton> resp_Triton = tpc::ELoss<Coll, Trks::iterator, PID::Triton>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Helium3> resp_Helium3 = tpc::ELoss<Coll, Trks::iterator, PID::Helium3>();
    constexpr tpc::ELoss<Coll, Trks::iterator, PID::Alpha> resp_Alpha = tpc::ELoss<Coll, Trks::iterator, PID::Alpha>();

    tpcpid.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tpcpid(
        resp_Electron.GetExpectedSignal(resp, collision, trk),
        resp_Muon.GetExpectedSignal(resp, collision, trk),
        resp_Pion.GetExpectedSignal(resp, collision, trk),
        resp_Kaon.GetExpectedSignal(resp, collision, trk),
        resp_Proton.GetExpectedSignal(resp, collision, trk),
        resp_Deuteron.GetExpectedSignal(resp, collision, trk),
        resp_Triton.GetExpectedSignal(resp, collision, trk),
        resp_Helium3.GetExpectedSignal(resp, collision, trk),
        resp_Alpha.GetExpectedSignal(resp, collision, trk),
        resp_Electron.GetExpectedSigma(resp, collision, trk),
        resp_Muon.GetExpectedSigma(resp, collision, trk),
        resp_Pion.GetExpectedSigma(resp, collision, trk),
        resp_Kaon.GetExpectedSigma(resp, collision, trk),
        resp_Proton.GetExpectedSigma(resp, collision, trk),
        resp_Deuteron.GetExpectedSigma(resp, collision, trk),
        resp_Triton.GetExpectedSigma(resp, collision, trk),
        resp_Helium3.GetExpectedSigma(resp, collision, trk),
        resp_Alpha.GetExpectedSigma(resp, collision, trk),
        resp_Electron.GetSeparation(resp, collision, trk),
        resp_Muon.GetSeparation(resp, collision, trk),
        resp_Pion.GetSeparation(resp, collision, trk),
        resp_Kaon.GetSeparation(resp, collision, trk),
        resp_Proton.GetSeparation(resp, collision, trk),
        resp_Deuteron.GetSeparation(resp, collision, trk),
        resp_Triton.GetSeparation(resp, collision, trk),
        resp_Helium3.GetSeparation(resp, collision, trk),
        resp_Alpha.GetSeparation(resp, collision, trk));
    }
  }
};

struct pidTPCTaskQA {
  static constexpr int Np = 9;
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr const char* hexpected[Np] = {"expected/El", "expected/Mu", "expected/Pi",
                                                "expected/Ka", "expected/Pr", "expected/De",
                                                "expected/Tr", "expected/He", "expected/Al"};
  static constexpr const char* hexpected_diff[Np] = {"expected_diff/El", "expected_diff/Mu", "expected_diff/Pi",
                                                     "expected_diff/Ka", "expected_diff/Pr", "expected_diff/De",
                                                     "expected_diff/Tr", "expected_diff/He", "expected_diff/Al"};
  static constexpr const char* hnsigma[Np] = {"nsigma/El", "nsigma/Mu", "nsigma/Pi",
                                              "nsigma/Ka", "nsigma/Pr", "nsigma/De",
                                              "nsigma/Tr", "nsigma/He", "nsigma/Al"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 20, "Maximum momentum in range"};

  void init(o2::framework::InitContext&)
  {

#define makelogaxis(h)                                            \
  {                                                               \
    const Int_t nbins = h->GetNbinsX();                           \
    double binp[nbins + 1];                                       \
    double max = h->GetXaxis()->GetBinUpEdge(nbins);              \
    double min = h->GetXaxis()->GetBinLowEdge(1);                 \
    if (min <= 0)                                                 \
      min = 0.00001;                                              \
    double lmin = TMath::Log10(min);                              \
    double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins); \
    for (int i = 0; i < nbins; i++) {                             \
      binp[i] = TMath::Exp(TMath::Log(10) * (lmin + i * ldelta)); \
    }                                                             \
    binp[nbins] = max + 1;                                        \
    h->GetXaxis()->Set(nbins, binp);                              \
  }

    // Event properties
    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", kTH1F, {{100, -20, 20}});
    histos.add("event/tpcsignal", ";#it{p} (GeV/#it{c});TPC Signal", kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 1000}});
    makelogaxis(histos.get<TH2>("event/tpcsignal"));
    for (int i = 0; i < Np; i++) {
      // Exp signal
      histos.add(hexpected[i], Form(";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_(%s)", pT[i]), kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 1000}});
      makelogaxis(histos.get<TH2>(hexpected[i]));
      // Signal - Expected signal
      histos.add(hexpected_diff[i], Form(";#it{p} (GeV/#it{c});;d#it{E}/d#it{x} - d#it{E}/d#it{x}(%s)", pT[i]), kTH2F, {{nBinsP, MinP, MaxP}, {1000, -500, 500}});
      makelogaxis(histos.get<TH2>(hexpected_diff[i]));
      // NSigma
      histos.add(hnsigma[i], Form(";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(%s)", pT[i]), kTH2F, {{nBinsP, MinP, MaxP}, {200, -10, 10}});
      makelogaxis(histos.get<TH2>(hnsigma[i]));
    }
#undef makelogaxis
  }
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC> const& tracks)
  {
    histos.fill("event/vertexz", collision.posZ());

    for (auto t : tracks) {
      // const float mom = t.p();
      const float mom = t.tpcInnerParam();
      histos.fill("event/tpcsignal", mom, t.tpcSignal());
      //
      const float exp[Np] = {t.tpcExpSignalEl(), t.tpcExpSignalMu(), t.tpcExpSignalPi(),
                             t.tpcExpSignalKa(), t.tpcExpSignalPr(), t.tpcExpSignalDe(),
                             t.tpcExpSignalTr(), t.tpcExpSignalHe(), t.tpcExpSignalAl()};
      for (int i = 0; i < Np; i++) {
        histos.fill(hexpected[i], mom, exp[i]);
        histos.fill(hexpected_diff[i], mom, t.tpcSignal() - exp[i]);
      }
      //
      const float nsigma[Np] = {t.tpcNSigmaEl(), t.tpcNSigmaMu(), t.tpcNSigmaPi(),
                                t.tpcNSigmaKa(), t.tpcNSigmaPr(), t.tpcNSigmaDe(),
                                t.tpcNSigmaTr(), t.tpcNSigmaHe(), t.tpcNSigmaAl()};
      for (int i = 0; i < Np; i++) {
        histos.fill(hnsigma[i], t.p(), nsigma[i]);
      }
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
