// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFMatcherSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"

// from Tracks
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"

// from TOF
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/TOFMatcherSpec.h"

// from FIT
#include "DataFormatsFT0/RecPoints.h"

#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;
// using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
// using GTrackID = o2::dataformats::GlobalTrackID;
// using DetID = o2::detectors::DetID;

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class TOFMatcherSpec : public Task
{
 public:
  TOFMatcherSpec(std::shared_ptr<DataRequest> dr, bool useMC, bool useFIT) : mDataRequest(dr), mUseMC(useMC), mUseFIT(useFIT) {}
  ~TOFMatcherSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  bool mUseMC = true;
  bool mUseFIT = false;
  MatchTOF mMatcher; ///< Cluster finder
  TStopwatch mTimer;
};

void TOFMatcherSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};

  // this is a hack to provide Mat.LUT from the local file, in general will be provided by the framework from CCDB
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::utils::Str::pathExists(matLUTFile)) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    LOG(INFO) << "Loaded material LUT from " << matLUTFile;
  } else {
    LOG(INFO) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
  }
}

void TOFMatcherSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  const auto clustersRO = pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster");

  if (mUseFIT) {
    // Note: the particular variable will go out of scope, but the span is passed by copy to the
    // worker and the underlying memory is valid throughout the whole computation
    auto recPoints = std::move(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("fitrecpoints"));
    mMatcher.setFITRecPoints(recPoints);
    LOG(INFO) << "TOF Reco Workflow pulled " << recPoints.size() << " FIT RecPoints";
  }

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> toflab;
  if (mUseMC) {
    const auto toflabel = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabel");
    toflab = std::move(*toflabel);
  }

  //mMatcher.run(tracksRO, clustersRO, toflab, itstpclab);

  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHINFOS", 0, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector());
  if (mUseMC) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MCMATCHTOF", 0, Lifetime::Timeframe}, mMatcher.getMatchedTOFLabelsVector());
  }
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe}, mMatcher.getCalibVector());

  mTimer.Stop();
}

void TOFMatcherSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TOF matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTOFMatcherSpec(GTrackID::mask_t src, bool useMC, bool useFIT)
{
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(src, useMC);

  std::vector<InputSpec> inputs = dataRequest->inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("tofcluster", o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("tofclusterlabel", o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }

  if (useFIT) {
    inputs.emplace_back("fitrecpoints", o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHINFOS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MCMATCHTOF", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tof-matcher",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFMatcherSpec>(dataRequest, useMC, useFIT)},
    Options{
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace globaltracking
} // namespace o2
