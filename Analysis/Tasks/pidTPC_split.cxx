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
    {"pid-el", VariantType::Int, 1, {"Produce PID information for the electron mass hypothesis"}},
    {"pid-mu", VariantType::Int, 1, {"Produce PID information for the muon mass hypothesis"}},
    {"pid-pikapr", VariantType::Int, 1, {"Produce PID information for the Pion, Kaon, Proton mass hypothesis"}},
    {"pid-nuclei", VariantType::Int, 1, {"Produce PID information for the Deuteron, Triton, Alpha mass hypothesis"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

template <o2::track::PID::ID pid_type, typename table>
struct pidTPCTaskPerParticle {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collisions;
  Produces<table> tpcpid;
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
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, pid_type> resp_PID = tpc::ELoss<Coll::iterator, Trks::iterator, pid_type>();

    tpcpid.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tpcpid(resp_PID.GetExpectedSigma(resp, trk.collision(), trk),
             resp_PID.GetSeparation(resp, trk.collision(), trk));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow;
  if (cfgc.options().get<int>("pid-el")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Electron, o2::aod::pidRespTPCEl>>("pidTPCEl-task"));
  }
  if (cfgc.options().get<int>("pid-mu")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Muon, o2::aod::pidRespTPCMu>>("pidTPCMu-task"));
  }
  if (cfgc.options().get<int>("pid-pikapr")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Pion, o2::aod::pidRespTPCPi>>("pidTPCPi-task"));
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Kaon, o2::aod::pidRespTPCKa>>("pidTPCKa-task"));
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Proton, o2::aod::pidRespTPCPr>>("pidTPCPr-task"));
  }
  if (cfgc.options().get<int>("pid-nuclei")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Deuteron, o2::aod::pidRespTPCDe>>("pidTPCDe-task"));
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Triton, o2::aod::pidRespTPCTr>>("pidTPCTr-task"));
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Helium3, o2::aod::pidRespTPCHe>>("pidTPCHe-task"));
    workflow.push_back(adaptAnalysisTask<pidTPCTaskPerParticle<PID::Alpha, o2::aod::pidRespTPCAl>>("pidTPCAl-task"));
  }
  return workflow;
}
