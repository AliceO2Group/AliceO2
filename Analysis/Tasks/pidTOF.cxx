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

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<pidTOFTask>("pidTOF-task")};
}
