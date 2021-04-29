// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackerSpec.cxx

#include "MFTWorkflow/TrackerSpec.h"

#include "MFTTracking/TrackerConfig.h"
#include "MFTTracking/ROframe.h"
#include "MFTTracking/IOUtils.h"
#include "MFTTracking/Tracker.h"
#include "MFTTracking/TrackCA.h"
#include "MFTBase/GeometryTGeo.h"

#include <vector>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{

void TrackerDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    o2::base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    o2::base::GeometryManager::loadGeometry();
    o2::mft::GeometryTGeo* geom = o2::mft::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                   o2::math_utils::TransformType::T2G));

    // tracking configuration parameters
    auto& mftTrackingParam = MFTTrackingParam::Instance();
    // create the tracker: set the B-field, the configuration and initialize
    mTracker = std::make_unique<o2::mft::Tracker>(mUseMC);
    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    mTracker->setBz(field->getBz(centerMFT));
    mTracker->initConfig(mftTrackingParam, true);
    mTracker->initialize();
  } else {
    throw std::runtime_error(o2::utils::Str::concat_string("Cannot retrieve GRP from the ", filename));
  }

  std::string dictPath = ic.options().get<std::string>("mft-dictionary-path");
  std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::MFT, dictPath, ".bin");
  if (o2::utils::Str::pathExists(dictFile)) {
    mDict.readBinaryFile(dictFile);
    LOG(INFO) << "Tracker running with a provided dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, Tracker expects cluster patterns";
  }
}

void TrackerDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  auto nTracksLTF = 0;
  auto nTracksCA = 0;

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"MFT", "TRACKSROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  LOG(INFO) << "MFTTracker pulled " << compClusters.size() << " compressed clusters in "
            << rofsinput.size() << " RO frames";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = mUseMC ? pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release() : nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mUseMC) {
    // get the array as read-only span, a snapshot of the object is sent forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(INFO) << labels->getIndexedSize() << " MC label objects , in "
              << mc2rofs.size() << " MC events";
  }

  //std::vector<o2::mft::TrackMFTExt> tracks;
  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"MFT", "TRACKCLSID", 0, Lifetime::Timeframe});
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackLabels;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> allTrackLabels;
  std::vector<o2::mft::TrackLTF> tracksLTF;
  std::vector<o2::mft::TrackCA> tracksCA;
  auto& allTracksMFT = pc.outputs().make<std::vector<o2::mft::TrackMFT>>(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe});

  std::uint32_t roFrame = 0;
  o2::mft::ROframe event(0);

  Bool_t continuous = mGRP->isDetContinuousReadOut("MFT");
  LOG(INFO) << "MFTTracker RO: continuous=" << continuous;

  // snippet to convert found tracks to final output tracks with separate cluster indices
  auto copyTracks = [&event](auto& tracks, auto& allTracks, auto& allClusIdx) {
    for (auto& trc : tracks) {
      trc.setExternalClusterIndexOffset(allClusIdx.size());
      int ncl = trc.getNumberOfPoints();
      for (int ic = 0; ic < ncl; ic++) {
        auto externalClusterID = trc.getExternalClusterIndex(ic);
        allClusIdx.push_back(externalClusterID);
      }
      allTracks.emplace_back(trc);
    }
  };

  gsl::span<const unsigned char>::iterator pattIt = patterns.begin();
  if (continuous) {
    for (auto& rof : rofs) {
      int nclUsed = ioutils::loadROFrameData(rof, event, compClusters, pattIt, mDict, labels, mTracker.get());
      if (nclUsed) {
        event.setROFrameId(roFrame);
        event.initialize();
        LOG(INFO) << "ROframe: " << roFrame << ", clusters loaded : " << nclUsed;
        mTracker->setROFrame(roFrame);
        mTracker->clustersToTracks(event);
        tracksLTF.swap(event.getTracksLTF());
        tracksCA.swap(event.getTracksCA());
        nTracksLTF += tracksLTF.size();
        nTracksCA += tracksCA.size();

        if (mUseMC) {
          mTracker->computeTracksMClabels(tracksLTF);
          mTracker->computeTracksMClabels(tracksCA);
          trackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
          allTrackLabels.mergeAtBack(trackLabels);
        }

        LOG(INFO) << "Found tracks LTF: " << tracksLTF.size();
        LOG(INFO) << "Found tracks CA: " << tracksCA.size();
        int first = allTracksMFT.size();
        int number = tracksLTF.size() + tracksCA.size();
        rof.setFirstEntry(first);
        rof.setNEntries(number);
        copyTracks(tracksLTF, allTracksMFT, allClusIdx);
        copyTracks(tracksCA, allTracksMFT, allClusIdx);
      }
      roFrame++;
    }
  }

  LOG(INFO) << "MFTTracker found " << nTracksLTF << " tracks LTF";
  LOG(INFO) << "MFTTracker found " << nTracksCA << " tracks CA";
  LOG(INFO) << "MFTTracker pushed " << allTracksMFT.size() << " tracks";

  if (mUseMC) {
    pc.outputs().snapshot(Output{"MFT", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"MFT", "TRACKSMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }
  mTimer.Stop();
}

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "MFT Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTrackerSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "MFT", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKSROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKCLSID", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "MFT", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "TRACKSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(useMC)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the output file"}},
      {"mft-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}}}};
}

} // namespace mft
} // namespace o2
