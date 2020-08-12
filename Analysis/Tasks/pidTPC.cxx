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
  tpc::Response resp = tpc::Response();
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

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    for (auto const& i : tracks) {
      resp.UpdateTrack(i.tpcInnerParam(), i.tpcSignal(), i.tpcNClsShared());
      tpcpid(
        resp.GetExpectedSignal(PID::Electron),
        resp.GetExpectedSignal(PID::Muon),
        resp.GetExpectedSignal(PID::Pion),
        resp.GetExpectedSignal(PID::Kaon),
        resp.GetExpectedSignal(PID::Proton),
        resp.GetExpectedSignal(PID::Deuteron),
        resp.GetExpectedSignal(PID::Triton),
        resp.GetExpectedSignal(PID::Helium3),
        resp.GetExpectedSignal(PID::Alpha),
        resp.GetExpectedSigma(PID::Electron),
        resp.GetExpectedSigma(PID::Muon),
        resp.GetExpectedSigma(PID::Pion),
        resp.GetExpectedSigma(PID::Kaon),
        resp.GetExpectedSigma(PID::Proton),
        resp.GetExpectedSigma(PID::Deuteron),
        resp.GetExpectedSigma(PID::Triton),
        resp.GetExpectedSigma(PID::Helium3),
        resp.GetExpectedSigma(PID::Alpha),
        resp.GetSeparation(PID::Electron),
        resp.GetSeparation(PID::Muon),
        resp.GetSeparation(PID::Pion),
        resp.GetSeparation(PID::Kaon),
        resp.GetSeparation(PID::Proton),
        resp.GetSeparation(PID::Deuteron),
        resp.GetSeparation(PID::Triton),
        resp.GetSeparation(PID::Helium3),
        resp.GetSeparation(PID::Alpha));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<pidTPCTask>("pidTPC-task")};
}
