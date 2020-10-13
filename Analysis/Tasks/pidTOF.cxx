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

// #define USE_REGISTRY
#ifdef USE_REGISTRY
#include "Framework/HistogramRegistry.h"
#endif

// ROOT includes
#include <TH1F.h>

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

struct pidTOFTask {
  Produces<aod::pidRespTOF> tofpid;
  Produces<aod::pidRespTOFbeta> tofpidbeta;
  DetectorResponse<tof::Response> resp = DetectorResponse<tof::Response>();
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
    resp.SetParameters(DetectorResponse<tof::Response>::kSigma, p);
    const std::string fname = paramfile.value;
    if (!fname.empty()) { // Loading the parametrization from file
      resp.LoadParamFromFile(fname.data(), sigmaname.value, DetectorResponse<tof::Response>::kSigma);
    } else { // Loading it from CCDB
      const std::string path = "Analysis/PID/TOF";
      resp.LoadParam(DetectorResponse<tof::Response>::kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
    }
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    tof::EventTime evt = tof::EventTime();
    evt.SetEvTime(0, collision.collisionTime());
    evt.SetEvTimeReso(0, collision.collisionTimeRes());
    evt.SetEvTimeMask(0, collision.collisionTimeMask());
    resp.SetEventTime(evt);

    tofpidbeta.reserve(tracks.size());
    tofpid.reserve(tracks.size());
    for (auto const& i : tracks) {
      resp.UpdateTrack(i.p(), i.tofExpMom() / tof::Response::kCSPEED, i.length(), i.tofSignal());
      tofpidbeta(resp.GetBeta(),
                 resp.GetBetaExpectedSigma(),
                 resp.GetExpectedBeta(PID::Electron),
                 resp.GetBetaExpectedSigma(),
                 resp.GetBetaNumberOfSigmas(PID::Electron));
      tofpid(
        resp.GetExpectedSignal(PID::Electron),
        resp.GetExpectedSignal(PID::Muon),
        resp.GetExpectedSignal(PID::Pion),
        resp.GetExpectedSignal(PID::Kaon),
        resp.GetExpectedSignal(PID::Proton),
        resp.GetExpectedSignal(PID::Deuteron),
        resp.GetExpectedSignal(PID::Triton),
        resp.GetExpectedSignal(PID::Helium3),
        resp.GetExpectedSignal(PID::Alpha),
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

struct pidTOFTaskQA {
  enum event_histo : uint8_t { vertexz,
                               signal,
                               tofbeta };
  enum Particle : uint8_t { El,
                            Mu,
                            Pi,
                            Ka,
                            Pr,
                            De,
                            Tr,
                            He,
                            Al };

#ifdef USE_REGISTRY
  // Event
  HistogramRegistry event{
    "event",
    true,
    {
      {"hvertexz", ";Vtx_{z} (cm);Entries", {HistogramType::kTH1F, {{100, -20, 20}}}},
      {"htofsignal", ";#it{p} (GeV/#it{c});TOF Signal", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}} //
      {"htofbeta", ";#it{p} (GeV/#it{c});TOF #beta", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 2}}}}        //
    }                                                                                                           //
  };
  // Exp signal
  HistogramRegistry expected{
    "expected",
    true,
    {
      {"hexpectedEl", ";#it{p} (GeV/#it{c});t_{exp e}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedMu", ";#it{p} (GeV/#it{c});t_{exp #mu}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedPi", ";#it{p} (GeV/#it{c});t_{exp #pi}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedKa", ";#it{p} (GeV/#it{c});t_{exp K}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedPr", ";#it{p} (GeV/#it{c});t_{exp p}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedDe", ";#it{p} (GeV/#it{c});t_{exp d}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedTr", ";#it{p} (GeV/#it{c});t_{exp t}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedHe", ";#it{p} (GeV/#it{c});t_{exp ^{3}He}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}},
      {"hexpectedAl", ";#it{p} (GeV/#it{c});t_{exp #alpha}", {HistogramType::kTH2F, {{100, 0, 5}, {100, 0, 10000}}}} //
    }                                                                                                                //
  };
  // T-Texp
  HistogramRegistry timediff{
    "timediff",
    true,
    {
      {"htimediffEl", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp e})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffMu", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #mu})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffPi", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #pi})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffKa", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp K})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffPr", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp p})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffDe", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp d})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffTr", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp t})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffHe", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp ^{3}He})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}},
      {"htimediffAl", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #alpha})", {HistogramType::kTH2F, {{100, 0, 5}, {100, -1000, 1000}}}} //
    }                                                                                                                               //
  };

  // NSigma
  HistogramRegistry nsigma{
    "nsigma",
    true,
    {
      {"hnsigmaEl", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(e)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaMu", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(#mu)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaPi", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(#pi)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaKa", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(K)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaPr", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(p)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaDe", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(d)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaTr", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(t)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaHe", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(^{3}He)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}},
      {"hnsigmaAl", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(#alpha)", {HistogramType::kTH2F, {{1000, 0.001, 20}, {200, -10, 10}}}} //
    }                                                                                                                             //
  };
#else
  // Event
  OutputObj<experimental::histhelpers::HistFolder> event{experimental::histhelpers::HistFolder("event"), OutputObjHandlingPolicy::QAObject};
  // Exp signal
  OutputObj<experimental::histhelpers::HistFolder> expected{experimental::histhelpers::HistFolder("expected"), OutputObjHandlingPolicy::QAObject};
  // T-Texp
  OutputObj<experimental::histhelpers::HistFolder> timediff{experimental::histhelpers::HistFolder("timediff"), OutputObjHandlingPolicy::QAObject};
  // NSigma
  OutputObj<experimental::histhelpers::HistFolder> nsigma{experimental::histhelpers::HistFolder("nsigma"), OutputObjHandlingPolicy::QAObject};
#endif

  void init(o2::framework::InitContext&)
  {
#ifndef USE_REGISTRY
    event->Add<vertexz>(new TH1F("hvertexz", ";Vtx_{z} (cm);Entries", 100, -20, 20));
    event->Add<signal>(new TH2F("htofsignal", ";#it{p} (GeV/#it{c});TOF Signal", 100, 0, 5, 100, 0, 10000));
    event->Add<tofbeta>(new TH2F("htofbeta", ";#it{p} (GeV/#it{c});TOF #beta", 100, 0, 5, 100, 0, 2));
    //
    expected->Add<El>(new TH2F("hexpectedEl", ";#it{p} (GeV/#it{c});t_{exp e}", 100, 0, 5, 100, 0, 10000));
    expected->Add<Mu>(new TH2F("hexpectedMu", ";#it{p} (GeV/#it{c});t_{exp #mu}", 100, 0, 5, 100, 0, 10000));
    expected->Add<Pi>(new TH2F("hexpectedPi", ";#it{p} (GeV/#it{c});t_{exp #pi}", 100, 0, 5, 100, 0, 10000));
    expected->Add<Ka>(new TH2F("hexpectedKa", ";#it{p} (GeV/#it{c});t_{exp K}", 100, 0, 5, 100, 0, 10000));
    expected->Add<Pr>(new TH2F("hexpectedPr", ";#it{p} (GeV/#it{c});t_{exp p}", 100, 0, 5, 100, 0, 10000));
    expected->Add<De>(new TH2F("hexpectedDe", ";#it{p} (GeV/#it{c});t_{exp d}", 100, 0, 5, 100, 0, 10000));
    expected->Add<Tr>(new TH2F("hexpectedTr", ";#it{p} (GeV/#it{c});t_{exp t}", 100, 0, 5, 100, 0, 10000));
    expected->Add<He>(new TH2F("hexpectedHe", ";#it{p} (GeV/#it{c});t_{exp ^{3}He}", 100, 0, 5, 100, 0, 10000));
    expected->Add<Al>(new TH2F("hexpectedAl", ";#it{p} (GeV/#it{c});t_{exp #alpha}", 100, 0, 5, 100, 0, 10000));
    //
    timediff->Add<El>(new TH2F("htimediffEl", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp e})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<Mu>(new TH2F("htimediffMu", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #mu})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<Pi>(new TH2F("htimediffPi", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #pi})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<Ka>(new TH2F("htimediffKa", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp K})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<Pr>(new TH2F("htimediffPr", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp p})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<De>(new TH2F("htimediffDe", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp d})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<Tr>(new TH2F("htimediffTr", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp t})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<He>(new TH2F("htimediffHe", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp ^{3}He})", 100, 0, 5, 100, -1000, 1000));
    timediff->Add<Al>(new TH2F("htimediffAl", ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #alpha})", 100, 0, 5, 100, -1000, 1000));
    //
    nsigma->Add<El>(new TH2F("nsigmaEl", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(e)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Mu>(new TH2F("nsigmaMu", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(#mu)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Pi>(new TH2F("nsigmaPi", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(#pi)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Ka>(new TH2F("nsigmaKa", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(K)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Pr>(new TH2F("nsigmaPr", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(p)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<De>(new TH2F("nsigmaDe", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(d)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Tr>(new TH2F("nsigmaTr", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(t)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<He>(new TH2F("nsigmaHe", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(^{3}He)", 1000, 0.001, 20, 200, -10, 10));
    nsigma->Add<Al>(new TH2F("nsigmaAl", ";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(#alpha)", 1000, 0.001, 20, 200, -10, 10));
#endif
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta> const& tracks)
  {
#ifdef USE_REGISTRY
    event("vertexz")->Fill(collision.posZ());
#else
    event->Fill<vertexz>(collision.posZ());
#endif

    for (auto i : tracks) {
      //
      const float tof = i.tofSignal() - collision.collisionTime();
#ifdef USE_REGISTRY
      event("htofsignal")->Fill(i.p(), i.tofSignal());
      event("htofbeta")->Fill(i.p(), i.beta());
      //
      expected("hexpectedEl")->Fill(i.p(), i.tofExpSignalEl());
      expected("hexpectedEl")->Fill(i.p(), i.tofExpSignalEl());
      expected("hexpectedMu")->Fill(i.p(), i.tofExpSignalMu());
      expected("hexpectedPi")->Fill(i.p(), i.tofExpSignalPi());
      expected("hexpectedKa")->Fill(i.p(), i.tofExpSignalKa());
      expected("hexpectedPr")->Fill(i.p(), i.tofExpSignalPr());
      expected("hexpectedDe")->Fill(i.p(), i.tofExpSignalDe());
      expected("hexpectedTr")->Fill(i.p(), i.tofExpSignalTr());
      expected("hexpectedHe")->Fill(i.p(), i.tofExpSignalHe());
      expected("hexpectedAl")->Fill(i.p(), i.tofExpSignalAl());
      //
      timediff("htimediffEl")->Fill(i.p(), tof - i.tofExpSignalEl());
      timediff("htimediffMu")->Fill(i.p(), tof - i.tofExpSignalMu());
      timediff("htimediffPi")->Fill(i.p(), tof - i.tofExpSignalPi());
      timediff("htimediffKa")->Fill(i.p(), tof - i.tofExpSignalKa());
      timediff("htimediffPr")->Fill(i.p(), tof - i.tofExpSignalPr());
      timediff("htimediffDe")->Fill(i.p(), tof - i.tofExpSignalDe());
      timediff("htimediffTr")->Fill(i.p(), tof - i.tofExpSignalTr());
      timediff("htimediffHe")->Fill(i.p(), tof - i.tofExpSignalHe());
      timediff("htimediffAl")->Fill(i.p(), tof - i.tofExpSignalAl());
      //
      nsigma("hnsigmaEl")->Fill(i.p(), i.tofNSigmaEl());
      nsigma("hnsigmaMu")->Fill(i.p(), i.tofNSigmaMu());
      nsigma("hnsigmaPi")->Fill(i.p(), i.tofNSigmaPi());
      nsigma("hnsigmaKa")->Fill(i.p(), i.tofNSigmaKa());
      nsigma("hnsigmaPr")->Fill(i.p(), i.tofNSigmaPr());
      nsigma("hnsigmaDe")->Fill(i.p(), i.tofNSigmaDe());
      nsigma("hnsigmaTr")->Fill(i.p(), i.tofNSigmaTr());
      nsigma("hnsigmaHe")->Fill(i.p(), i.tofNSigmaHe());
      nsigma("hnsigmaAl")->Fill(i.p(), i.tofNSigmaAl());
#else
      event->Fill<signal>(i.p(), i.tofSignal());
      event->Fill<tofbeta>(i.p(), i.beta());
      //
      expected->Fill<El>(i.p(), i.tofExpSignalEl());
      expected->Fill<Mu>(i.p(), i.tofExpSignalMu());
      expected->Fill<Pi>(i.p(), i.tofExpSignalPi());
      expected->Fill<Ka>(i.p(), i.tofExpSignalKa());
      expected->Fill<Pr>(i.p(), i.tofExpSignalPr());
      expected->Fill<De>(i.p(), i.tofExpSignalDe());
      expected->Fill<Tr>(i.p(), i.tofExpSignalTr());
      expected->Fill<He>(i.p(), i.tofExpSignalHe());
      expected->Fill<Al>(i.p(), i.tofExpSignalAl());
      //
      timediff->Fill<El>(i.p(), tof - i.tofExpSignalEl());
      timediff->Fill<Mu>(i.p(), tof - i.tofExpSignalMu());
      timediff->Fill<Pi>(i.p(), tof - i.tofExpSignalPi());
      timediff->Fill<Ka>(i.p(), tof - i.tofExpSignalKa());
      timediff->Fill<Pr>(i.p(), tof - i.tofExpSignalPr());
      timediff->Fill<De>(i.p(), tof - i.tofExpSignalDe());
      timediff->Fill<Tr>(i.p(), tof - i.tofExpSignalTr());
      timediff->Fill<He>(i.p(), tof - i.tofExpSignalHe());
      timediff->Fill<Al>(i.p(), tof - i.tofExpSignalAl());
      //
      nsigma->Fill<El>(i.p(), i.tofNSigmaEl());
      nsigma->Fill<Mu>(i.p(), i.tofNSigmaMu());
      nsigma->Fill<Pi>(i.p(), i.tofNSigmaPi());
      nsigma->Fill<Ka>(i.p(), i.tofNSigmaKa());
      nsigma->Fill<Pr>(i.p(), i.tofNSigmaPr());
      nsigma->Fill<De>(i.p(), i.tofNSigmaDe());
      nsigma->Fill<Tr>(i.p(), i.tofNSigmaTr());
      nsigma->Fill<He>(i.p(), i.tofNSigmaHe());
      nsigma->Fill<Al>(i.p(), i.tofNSigmaAl());
#endif
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{adaptAnalysisTask<pidTOFTask>("pidTOF-task")};
  if (cfgc.options().get<int>("add-qa")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA>("pidTOFQA-task"));
  }
  return workflow;
}
