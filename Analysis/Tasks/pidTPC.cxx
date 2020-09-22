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
#include "ReconstructionDataFormats/Track.h"
#include "PID/PIDResponse.h"
#include <CCDB/BasicCCDBManager.h>
#include "Analysis/HistHelpers.h"

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
  Produces<aod::pidRespTPC> tpcpid;
  DetectorResponse<tpc::Response> resp = DetectorResponse<tpc::Response>();
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
      resp.LoadParamFromFile(fname.data(), signalname.value, DetectorResponse<tpc::Response>::kSignal);
      resp.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse<tpc::Response>::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TPC";
      resp.LoadParam(DetectorResponse<tpc::Response>::kSignal, ccdb->getForTimeStamp<Parametrization>(path + "/" + signalname.value, timestamp.value));
      resp.LoadParam(DetectorResponse<tpc::Response>::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
    }
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    tpcpid.reserve(tracks.size());
    for (auto const& i : tracks) {
      resp.UpdateTrack(i.tpcInnerParam(), i.tpcSignal(), i.tpcNClsShared());
      tpcpid(
        resp.GetExpectedSignal(resp, PID::Electron),
        resp.GetExpectedSignal(resp, PID::Muon),
        resp.GetExpectedSignal(resp, PID::Pion),
        resp.GetExpectedSignal(resp, PID::Kaon),
        resp.GetExpectedSignal(resp, PID::Proton),
        resp.GetExpectedSignal(resp, PID::Deuteron),
        resp.GetExpectedSignal(resp, PID::Triton),
        resp.GetExpectedSignal(resp, PID::Helium3),
        resp.GetExpectedSignal(resp, PID::Alpha),
        resp.GetExpectedSigma(resp, PID::Electron),
        resp.GetExpectedSigma(resp, PID::Muon),
        resp.GetExpectedSigma(resp, PID::Pion),
        resp.GetExpectedSigma(resp, PID::Kaon),
        resp.GetExpectedSigma(resp, PID::Proton),
        resp.GetExpectedSigma(resp, PID::Deuteron),
        resp.GetExpectedSigma(resp, PID::Triton),
        resp.GetExpectedSigma(resp, PID::Helium3),
        resp.GetExpectedSigma(resp, PID::Alpha),
        resp.GetSeparation(resp, PID::Electron),
        resp.GetSeparation(resp, PID::Muon),
        resp.GetSeparation(resp, PID::Pion),
        resp.GetSeparation(resp, PID::Kaon),
        resp.GetSeparation(resp, PID::Proton),
        resp.GetSeparation(resp, PID::Deuteron),
        resp.GetSeparation(resp, PID::Triton),
        resp.GetSeparation(resp, PID::Helium3),
        resp.GetSeparation(resp, PID::Alpha));
    }
  }
};

struct pidTPCTaskQA {
  enum event_histo : uint8_t { vertexz,
                               signal };
  enum Particle : uint8_t { El,
                            Mu,
                            Pi,
                            Ka,
                            Pr,
                            De,
                            Tr,
                            He,
                            Al
  };

  // Event
  OutputObj<experimental::histhelpers::HistFolder> event{experimental::histhelpers::HistFolder("event"), OutputObjHandlingPolicy::QAObject};
  // Exp signal
  OutputObj<experimental::histhelpers::HistFolder> expected{experimental::histhelpers::HistFolder("expected"), OutputObjHandlingPolicy::QAObject};
  // Exp signal difference
  OutputObj<experimental::histhelpers::HistFolder> expected_diff{experimental::histhelpers::HistFolder("expected_diff"), OutputObjHandlingPolicy::QAObject};
  // NSigma
  OutputObj<experimental::histhelpers::HistFolder> nsigma{experimental::histhelpers::HistFolder("nsigma"), OutputObjHandlingPolicy::QAObject};
  void init(o2::framework::InitContext&)
  {

#define makelogaxis(h)                                            \
  {                                                               \
    const Int_t nbins = h->GetNbinsX();                           \
    double binp[nbins + 1];                                       \
    double max = h->GetXaxis()->GetBinUpEdge(nbins);              \
    double min = h->GetXaxis()->GetBinLowEdge(1);                 \
    double lmin = TMath::Log10(min);                              \
    double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins); \
    for (int i = 0; i < nbins; i++) {                             \
      binp[i] = TMath::Exp(TMath::Log(10) * (lmin + i * ldelta)); \
    }                                                             \
    binp[nbins] = max + 1;                                        \
    h->GetXaxis()->Set(nbins, binp);                              \
  }

    event->Add<vertexz>(new TH1F("hvertexz", ";Vtx_{z} (cm);Entries", 100, -20, 20));
    event->Add<signal>(new TH2F("htpcsignal", ";#it{p} (GeV/#it{c});TPC Signal", 1000, 0.001, 20, 1000, 0, 1000));
    makelogaxis(event->Get<TH2>(signal));
    //
    expected->Add<El>(new TH2F("hexpectedEl", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{e}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<Mu>(new TH2F("hexpectedMu", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{#mu}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<Pi>(new TH2F("hexpectedPi", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{#pi}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<Ka>(new TH2F("hexpectedKa", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{K}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<Pr>(new TH2F("hexpectedPr", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{p}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<De>(new TH2F("hexpectedDe", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{d}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<Tr>(new TH2F("hexpectedTr", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{t}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<He>(new TH2F("hexpectedHe", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{^{3}He}", 1000, 0.001, 20, 1000, 0, 1000));
    expected->Add<Al>(new TH2F("hexpectedAl", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{#alpha}", 1000, 0.001, 20, 1000, 0, 1000));
    makelogaxis(expected->Get<TH2>(El));
    makelogaxis(expected->Get<TH2>(Mu));
    makelogaxis(expected->Get<TH2>(Pi));
    makelogaxis(expected->Get<TH2>(Ka));
    makelogaxis(expected->Get<TH2>(Pr));
    makelogaxis(expected->Get<TH2>(De));
    makelogaxis(expected->Get<TH2>(Tr));
    makelogaxis(expected->Get<TH2>(He));
    makelogaxis(expected->Get<TH2>(Al));
    //
    expected_diff->Add<El>(new TH2F("hexpdiffEl", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{e}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<Mu>(new TH2F("hexpdiffMu", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{#mu}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<Pi>(new TH2F("hexpdiffPi", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{#pi}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<Ka>(new TH2F("hexpdiffKa", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{K}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<Pr>(new TH2F("hexpdiffPr", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{p}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<De>(new TH2F("hexpdiffDe", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{d}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<Tr>(new TH2F("hexpdiffTr", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{t}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<He>(new TH2F("hexpdiffHe", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{^{3}He}", 1000, 0.001, 20, 1000, -500, 500));
    expected_diff->Add<Al>(new TH2F("hexpdiffAl", ";#it{p} (GeV/#it{c});d#it{E}/d#it{x} - d#it{E}/d#it{x}_{#alpha}", 1000, 0.001, 20, 1000, -500, 500));
    makelogaxis(expected_diff->Get<TH2>(El));
    makelogaxis(expected_diff->Get<TH2>(Mu));
    makelogaxis(expected_diff->Get<TH2>(Pi));
    makelogaxis(expected_diff->Get<TH2>(Ka));
    makelogaxis(expected_diff->Get<TH2>(Pr));
    makelogaxis(expected_diff->Get<TH2>(De));
    makelogaxis(expected_diff->Get<TH2>(Tr));
    makelogaxis(expected_diff->Get<TH2>(He));
    makelogaxis(expected_diff->Get<TH2>(Al));
    //
    nsigma->Add<El>(new TH2F("hnsigmaEl", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(e)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Mu>(new TH2F("hnsigmaMu", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(#mu)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Pi>(new TH2F("hnsigmaPi", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(#pi)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Ka>(new TH2F("hnsigmaKa", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(K)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Pr>(new TH2F("hnsigmaPr", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(p)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<De>(new TH2F("hnsigmaDe", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(d)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Tr>(new TH2F("hnsigmaTr", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(t)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<He>(new TH2F("hnsigmaHe", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(^{3}He)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Al>(new TH2F("hnsigmaAl", ";#it{p} (GeV/#it{c});N_{#sigma}^{TPC}(#alpha)", 1000, 0.001, 20, 200, -10, 10));
    makelogaxis(nsigma->Get<TH2>(El));
    makelogaxis(nsigma->Get<TH2>(Mu));
    makelogaxis(nsigma->Get<TH2>(Pi));
    makelogaxis(nsigma->Get<TH2>(Ka));
    makelogaxis(nsigma->Get<TH2>(Pr));
    makelogaxis(nsigma->Get<TH2>(De));
    makelogaxis(nsigma->Get<TH2>(Tr));
    makelogaxis(nsigma->Get<TH2>(He));
    makelogaxis(nsigma->Get<TH2>(Al));
#undef makelogaxis
  }
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC> const& tracks)
  {
    event->Fill<vertexz>(collision.posZ());
    for (auto i : tracks) {
      // const float mom = i.p();
      const float mom = i.tpcInnerParam();
      event->Fill<signal>(mom, i.tpcSignal());
      //
      expected->Fill<El>(mom, i.tpcExpSignalEl());
      expected->Fill<Mu>(mom, i.tpcExpSignalMu());
      expected->Fill<Pi>(mom, i.tpcExpSignalPi());
      expected->Fill<Ka>(mom, i.tpcExpSignalKa());
      expected->Fill<Pr>(mom, i.tpcExpSignalPr());
      expected->Fill<De>(mom, i.tpcExpSignalDe());
      expected->Fill<Tr>(mom, i.tpcExpSignalTr());
      expected->Fill<He>(mom, i.tpcExpSignalHe());
      expected->Fill<Al>(mom, i.tpcExpSignalAl());
      //
      expected_diff->Fill<El>(mom, i.tpcSignal() - i.tpcExpSignalEl());
      expected_diff->Fill<Mu>(mom, i.tpcSignal() - i.tpcExpSignalMu());
      expected_diff->Fill<Pi>(mom, i.tpcSignal() - i.tpcExpSignalPi());
      expected_diff->Fill<Ka>(mom, i.tpcSignal() - i.tpcExpSignalKa());
      expected_diff->Fill<Pr>(mom, i.tpcSignal() - i.tpcExpSignalPr());
      expected_diff->Fill<De>(mom, i.tpcSignal() - i.tpcExpSignalDe());
      expected_diff->Fill<Tr>(mom, i.tpcSignal() - i.tpcExpSignalTr());
      expected_diff->Fill<He>(mom, i.tpcSignal() - i.tpcExpSignalHe());
      expected_diff->Fill<Al>(mom, i.tpcSignal() - i.tpcExpSignalAl());
      //
      nsigma->Fill<El>(mom, i.tpcNSigmaEl());
      nsigma->Fill<Mu>(mom, i.tpcNSigmaMu());
      nsigma->Fill<Pi>(mom, i.tpcNSigmaPi());
      nsigma->Fill<Ka>(mom, i.tpcNSigmaKa());
      nsigma->Fill<Pr>(mom, i.tpcNSigmaPr());
      nsigma->Fill<De>(mom, i.tpcNSigmaDe());
      nsigma->Fill<Tr>(mom, i.tpcNSigmaTr());
      nsigma->Fill<He>(mom, i.tpcNSigmaHe());
      nsigma->Fill<Al>(mom, i.tpcNSigmaAl());
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
