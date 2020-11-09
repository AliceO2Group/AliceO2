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
#include "PID/PIDResponse.h"
#include "PID/PIDTOF.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"add-qa", VariantType::Int, 0, {"Produce TOF PID QA histograms"}},
    {"add-beta", VariantType::Int, 0, {"Produce TOF Beta table"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct pidTOFTask {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collision;
  Produces<aod::pidRespTOF> tofpid;
  DetectorResponse resp;
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
    resp.SetParameters(DetectorResponse::kSigma, p);
    const std::string fname = paramfile.value;
    if (!fname.empty()) { // Loading the parametrization from file
      resp.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TOF";
      resp.LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
    }
  }

  void process(Coll const& collision, Trks const& tracks)
  {
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Electron> resp_Electron = tof::ExpTimes<Coll, Trks::iterator, PID::Electron>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Muon> resp_Muon = tof::ExpTimes<Coll, Trks::iterator, PID::Muon>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Pion> resp_Pion = tof::ExpTimes<Coll, Trks::iterator, PID::Pion>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Kaon> resp_Kaon = tof::ExpTimes<Coll, Trks::iterator, PID::Kaon>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Proton> resp_Proton = tof::ExpTimes<Coll, Trks::iterator, PID::Proton>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Deuteron> resp_Deuteron = tof::ExpTimes<Coll, Trks::iterator, PID::Deuteron>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Triton> resp_Triton = tof::ExpTimes<Coll, Trks::iterator, PID::Triton>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Helium3> resp_Helium3 = tof::ExpTimes<Coll, Trks::iterator, PID::Helium3>();
    constexpr tof::ExpTimes<Coll, Trks::iterator, PID::Alpha> resp_Alpha = tof::ExpTimes<Coll, Trks::iterator, PID::Alpha>();

    tofpid.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tofpid(resp_Electron.GetExpectedSignal(collision, trk),
             resp_Muon.GetExpectedSignal(collision, trk),
             resp_Pion.GetExpectedSignal(collision, trk),
             resp_Kaon.GetExpectedSignal(collision, trk),
             resp_Proton.GetExpectedSignal(collision, trk),
             resp_Deuteron.GetExpectedSignal(collision, trk),
             resp_Triton.GetExpectedSignal(collision, trk),
             resp_Helium3.GetExpectedSignal(collision, trk),
             resp_Alpha.GetExpectedSignal(collision, trk),
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

struct pidTOFTaskBeta {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collision;
  Produces<aod::pidRespTOFbeta> tofpidbeta;
  tof::Beta<Coll, Trks::iterator, PID::Electron> resp_Electron;
  Configurable<float> expreso{"tof-expreso", 80, "Expected resolution for the computation of the expected beta"};

  void init(o2::framework::InitContext&)
  {
    resp_Electron.mExpectedResolution = expreso.value;
  }

  void process(Coll const& collision, Trks const& tracks)
  {
    tofpidbeta.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tofpidbeta(resp_Electron.GetBeta(collision, trk),
                 resp_Electron.GetExpectedSigma(collision, trk),
                 resp_Electron.GetExpectedSignal(collision, trk),
                 resp_Electron.GetExpectedSigma(collision, trk),
                 resp_Electron.GetSeparation(collision, trk));
    }
  }
};

struct pidTOFTaskQA {
  static constexpr int Np = 9;
  static constexpr const char* hexpected[Np] = {"expected/El", "expected/Mu", "expected/Pi",
                                                "expected/Ka", "expected/Pr", "expected/De",
                                                "expected/Tr", "expected/He", "expected/Al"};
  static constexpr const char* hexpected_diff[Np] = {"expected_diff/El", "expected_diff/Mu", "expected_diff/Pi",
                                                     "expected_diff/Ka", "expected_diff/Pr", "expected_diff/De",
                                                     "expected_diff/Tr", "expected_diff/He", "expected_diff/Al"};
  static constexpr const char* hnsigma[Np] = {"nsigma/El", "nsigma/Mu", "nsigma/Pi",
                                              "nsigma/Ka", "nsigma/Pr", "nsigma/De",
                                              "nsigma/Tr", "nsigma/He", "nsigma/Al"};
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0.1, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 5, "Maximum momentum in range"};

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

  void init(o2::framework::InitContext&)
  {
    // Event properties
    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", HistType::kTH1F, {{100, -20, 20}});
    histos.add("event/colltime", ";Collision time (ps);Entries", HistType::kTH1F, {{100, -2000, 2000}});
    histos.add("event/tofsignal", ";#it{p} (GeV/#it{c});TOF Signal", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {10000, 0, 2e6}});
    makelogaxis(histos.get<TH2>("event/tofsignal"));
    histos.add("event/tofbeta", ";#it{p} (GeV/#it{c});TOF #beta", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 2}});
    makelogaxis(histos.get<TH2>("event/tofbeta"));
    for (int i = 0; i < Np; i++) {
      // Exp signal
      histos.add(hexpected[i], Form(";#it{p} (GeV/#it{c});t_{exp}(%s)", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 2e6}});
      makelogaxis(histos.get<TH2>(hexpected[i]));
      // T-Texp
      histos.add(hexpected_diff[i], Form(";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp}(%s))", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {100, -1000, 1000}});
      makelogaxis(histos.get<TH2>(hexpected_diff[i]));
      // NSigma
      histos.add(hnsigma[i], Form(";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {200, -10, 10}});
      makelogaxis(histos.get<TH2>(hnsigma[i]));
    }
  }
#undef makelogaxis

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta> const& tracks)
  {
    histos.fill("event/vertexz", collision.posZ());
    histos.fill("event/colltime", collision.collisionTime());

    for (auto t : tracks) {
      //
      if (t.tofSignal() < 0) { // Skipping tracks without TOF
        continue;
      }
      const float tof = t.tofSignal() - collision.collisionTime();
      //
      histos.fill("event/tofsignal", t.p(), t.tofSignal());
      histos.fill("event/tofbeta", t.p(), t.beta());
      //
      const float exp[Np] = {t.tofExpSignalEl(), t.tofExpSignalMu(), t.tofExpSignalPi(),
                             t.tofExpSignalKa(), t.tofExpSignalPr(), t.tofExpSignalDe(),
                             t.tofExpSignalTr(), t.tofExpSignalHe(), t.tofExpSignalAl()};
      for (int i = 0; i < Np; i++) {
        histos.fill(hexpected[i], t.p(), exp[i]);
        histos.fill(hexpected_diff[i], t.p(), tof - exp[i]);
      }
      //
      const float nsigma[Np] = {t.tofNSigmaEl(), t.tofNSigmaMu(), t.tofNSigmaPi(),
                                t.tofNSigmaKa(), t.tofNSigmaPr(), t.tofNSigmaDe(),
                                t.tofNSigmaTr(), t.tofNSigmaHe(), t.tofNSigmaAl()};
      for (int i = 0; i < Np; i++) {
        histos.fill(hnsigma[i], t.p(), nsigma[i]);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{adaptAnalysisTask<pidTOFTask>("pidTOF-task")};
  const int add_beta = cfgc.options().get<int>("add-beta");
  const int add_qa = cfgc.options().get<int>("add-qa");
  if (add_beta || add_qa) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskBeta>("pidTOFBeta-task"));
  }
  if (add_qa) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA>("pidTOFQA-task"));
  }
  return workflow;
}
