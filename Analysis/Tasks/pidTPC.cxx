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
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "PID/PIDResponse.h"
#include "Framework/ASoAHelpers.h"
#include <CCDB/BasicCCDBManager.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

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

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<pidTPCTask>("pidTPC-task")};
}
