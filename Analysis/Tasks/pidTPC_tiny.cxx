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
    {"pid-all", VariantType::Int, 1, {"Produce PID information for all mass hypotheses"}},
    {"pid-el", VariantType::Int, 0, {"Produce PID information for the electron mass hypothesis"}},
    {"pid-mu", VariantType::Int, 0, {"Produce PID information for the muon mass hypothesis"}},
    {"pid-pikapr", VariantType::Int, 0, {"Produce PID information for the Pion, Kaon, Proton mass hypothesis"}},
    {"pid-nuclei", VariantType::Int, 0, {"Produce PID information for the Deuteron, Triton, Alpha mass hypothesis"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

template <o2::track::PID::ID pid_type, typename table>
struct pidTPCTaskTiny {
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
      const float exp_sigma = resp_PID.GetExpectedSigma(resp, trk.collision(), trk);
      const float separation = resp_PID.GetSeparation(resp, trk.collision(), trk);
      if (separation <= o2::aod::pidtpc_tiny::binned_min) {
        tpcpid(o2::aod::pidtpc_tiny::lower_bin);
      } else if (separation >= o2::aod::pidtpc_tiny::binned_max) {
        tpcpid(o2::aod::pidtpc_tiny::upper_bin);
      } else if (separation >= 0) {
        tpcpid(separation / o2::aod::pidtpc_tiny::bin_width + 0.5f);
      } else {
        tpcpid(separation / o2::aod::pidtpc_tiny::bin_width - 0.5f);
      }
    }
  }
};

struct pidTPCTaskTinyFull {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collisions;

  Produces<o2::aod::pidRespTPCTEl> tpcpidEl;
  Produces<o2::aod::pidRespTPCTMu> tpcpidMu;
  Produces<o2::aod::pidRespTPCTPi> tpcpidPi;
  Produces<o2::aod::pidRespTPCTKa> tpcpidKa;
  Produces<o2::aod::pidRespTPCTPr> tpcpidPr;
  Produces<o2::aod::pidRespTPCTDe> tpcpidDe;
  Produces<o2::aod::pidRespTPCTTr> tpcpidTr;
  Produces<o2::aod::pidRespTPCTHe> tpcpidHe;
  Produces<o2::aod::pidRespTPCTAl> tpcpidAl;

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

    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Electron> resp_El = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Electron>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Muon> resp_Mu = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Muon>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Pion> resp_Pi = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Pion>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Kaon> resp_Ka = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Kaon>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Proton> resp_Pr = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Proton>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Deuteron> resp_De = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Deuteron>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Triton> resp_Tr = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Triton>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Helium3> resp_He = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Helium3>();
    constexpr tpc::ELoss<Coll::iterator, Trks::iterator, PID::Alpha> resp_Al = tpc::ELoss<Coll::iterator, Trks::iterator, PID::Alpha>();

    tpcpidEl.reserve(tracks.size());
    tpcpidMu.reserve(tracks.size());
    tpcpidPi.reserve(tracks.size());
    tpcpidKa.reserve(tracks.size());
    tpcpidPr.reserve(tracks.size());
    tpcpidDe.reserve(tracks.size());
    tpcpidTr.reserve(tracks.size());
    tpcpidHe.reserve(tracks.size());
    tpcpidAl.reserve(tracks.size());
    for (auto const& trk : tracks) {
#define FILL_PID_TABLE(PID_TABLE, PID_RESPONSE)                                        \
  {                                                                                    \
    const float exp_sigma = PID_RESPONSE.GetExpectedSigma(resp, trk.collision(), trk); \
    const float separation = PID_RESPONSE.GetSeparation(resp, trk.collision(), trk);   \
    if (separation <= o2::aod::pidtpc_tiny::binned_min) {                              \
      PID_TABLE(o2::aod::pidtpc_tiny::lower_bin);                                      \
    } else if (separation >= o2::aod::pidtpc_tiny::binned_max) {                       \
      PID_TABLE(o2::aod::pidtpc_tiny::upper_bin);                                      \
    } else if (separation >= 0) {                                                      \
      PID_TABLE(separation / o2::aod::pidtpc_tiny::bin_width + 0.5f);                  \
    } else {                                                                           \
      PID_TABLE(separation / o2::aod::pidtpc_tiny::bin_width - 0.5f);                  \
    }                                                                                  \
  }

      FILL_PID_TABLE(tpcpidEl, resp_El);
      FILL_PID_TABLE(tpcpidMu, resp_Mu);
      FILL_PID_TABLE(tpcpidPi, resp_Pi);
      FILL_PID_TABLE(tpcpidKa, resp_Ka);
      FILL_PID_TABLE(tpcpidPr, resp_Pr);
      FILL_PID_TABLE(tpcpidDe, resp_De);
      FILL_PID_TABLE(tpcpidTr, resp_Tr);
      FILL_PID_TABLE(tpcpidHe, resp_He);
      FILL_PID_TABLE(tpcpidAl, resp_Al);
#undef FILL_PID_TABLE
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow;
  if (cfgc.options().get<int>("pid-all")) {
    workflow.push_back(adaptAnalysisTask<pidTPCTaskTinyFull>("pidTPCFull-task"));
  } else {
    if (cfgc.options().get<int>("pid-el")) {
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Electron, o2::aod::pidRespTPCTEl>>("pidTPCEl-task"));
    }
    if (cfgc.options().get<int>("pid-mu")) {
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Muon, o2::aod::pidRespTPCTMu>>("pidTPCMu-task"));
    }
    if (cfgc.options().get<int>("pid-pikapr")) {
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Pion, o2::aod::pidRespTPCTPi>>("pidTPCPi-task"));
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Kaon, o2::aod::pidRespTPCTKa>>("pidTPCKa-task"));
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Proton, o2::aod::pidRespTPCTPr>>("pidTPCPr-task"));
    }
    if (cfgc.options().get<int>("pid-nuclei")) {
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Deuteron, o2::aod::pidRespTPCTDe>>("pidTPCDe-task"));
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Triton, o2::aod::pidRespTPCTTr>>("pidTPCTr-task"));
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Helium3, o2::aod::pidRespTPCTHe>>("pidTPCHe-task"));
      workflow.push_back(adaptAnalysisTask<pidTPCTaskTiny<PID::Alpha, o2::aod::pidRespTPCTAl>>("pidTPCAl-task"));
    }
  }
  return workflow;
}
