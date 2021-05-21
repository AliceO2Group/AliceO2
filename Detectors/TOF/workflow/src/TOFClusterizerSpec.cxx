// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFWorkflowUtils/TOFClusterizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "TOFReconstruction/Clusterer.h"
#include "TOFReconstruction/CosmicProcessor.h"
#include "TOFReconstruction/DataReader.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsTOF/CosmicInfo.h"
#include "DataFormatsTOF/CalibInfoCluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "TOFCalibration/CalibTOFapi.h"
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"

#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

// RSTODO to remove once the framework will start propagating the header.firstTForbit
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::framework;
using namespace o2::dataformats;

namespace o2
{
namespace tof
{

// use the tasking system of DPL
// just need to implement 2 special methods init + run (there is no need to inherit from anything)
class TOFDPLClustererTask
{
  bool mUseMC = true;
  bool mUseCCDB = false;
  bool mIsCalib = false;
  bool mIsCosmic = false;
  int mTimeWin = 5000;

 public:
  explicit TOFDPLClustererTask(bool useMC, bool useCCDB, bool doCalib, bool isCosmic) : mUseMC(useMC), mUseCCDB(useCCDB), mIsCalib(doCalib), mIsCosmic(isCosmic) {}
  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
    mTimer.Stop();
    mTimer.Reset();

    mTimeWin = ic.options().get<int>("cluster-time-window");
    LOG(INFO) << "Is calibration from cluster on? " << mIsCalib;
    LOG(INFO) << "DeltaTime for clusterization = " << mTimeWin << " ps";
    LOG(INFO) << "Is cosmics? " << mIsCosmic;

    mClusterer.setCalibFromCluster(mIsCalib);
    mClusterer.setDeltaTforClustering(mTimeWin);
  }

  void run(framework::ProcessingContext& pc)
  {
    mTimer.Start(false);
    // get digit data
    auto digits = pc.inputs().get<gsl::span<o2::tof::Digit>>("tofdigits");
    auto row = pc.inputs().get<gsl::span<o2::tof::ReadoutWindowData>>("readoutwin");

    const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
    mClusterer.setFirstOrbit(dh->firstTForbit);

    auto labelvector = std::make_shared<std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>>();
    if (mUseMC) {
      auto digitlabels = pc.inputs().get<std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>*>("tofdigitlabels");
      *labelvector.get() = std::move(*digitlabels);
      mClusterer.setMCTruthContainer(&mClsLabels);
      mClsLabels.clear();
    }

    o2::dataformats::CalibLHCphaseTOF lhcPhaseObj;
    o2::dataformats::CalibTimeSlewingParamTOF channelCalibObj;

    if (mUseCCDB) { // read calibration objects from ccdb
      // check LHC phase
      auto lhcPhase = pc.inputs().get<o2::dataformats::CalibLHCphaseTOF*>("tofccdbLHCphase");
      auto channelCalib = pc.inputs().get<o2::dataformats::CalibTimeSlewingParamTOF*>("tofccdbChannelCalib");

      o2::dataformats::CalibLHCphaseTOF lhcPhaseObjTmp = std::move(*lhcPhase);
      o2::dataformats::CalibTimeSlewingParamTOF channelCalibObjTmp = std::move(*channelCalib);

      // make a copy in global scope
      lhcPhaseObj = lhcPhaseObjTmp;
      channelCalibObj = channelCalibObjTmp;
    } else { // calibration objects set to zero
      lhcPhaseObj.addLHCphase(0, 0);
      lhcPhaseObj.addLHCphase(2000000000, 0);

      for (int ich = 0; ich < o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELS; ich++) {
        channelCalibObj.addTimeSlewingInfo(ich, 0, 0);
        int sector = ich / o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
        int channelInSector = ich % o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
        channelCalibObj.setFractionUnderPeak(sector, channelInSector, 1);
      }
    }

    o2::tof::CalibTOFapi calibapi(long(0), &lhcPhaseObj, &channelCalibObj);

    mClusterer.setCalibApi(&calibapi);

    // call actual clustering routine
    mClustersArray.clear();
    if (mIsCalib) {
      mClusterer.getInfoFromCluster()->clear();
    }

    if (mIsCosmic) {
      mCosmicProcessor.clear();
    }

    for (int i = 0; i < row.size(); i++) {
      //printf("# TOF readout window for clusterization = %d/%lu (N digits = %d)\n", i, row.size(), row[i].size());
      auto digitsRO = row[i].getBunchChannelData(digits);
      mReader.setDigitArray(&digitsRO);

      if (mIsCosmic) {
        mCosmicProcessor.process(mReader, i != 0);
      }

      if (mUseMC) {
        mClusterer.process(mReader, mClustersArray, &(labelvector->at(i)));
      } else {
        mClusterer.process(mReader, mClustersArray, nullptr);
      }
    }
    LOG(DEBUG) << "TOF CLUSTERER : TRANSFORMED " << digits.size()
               << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send clusters
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe}, mClustersArray);
    // send labels
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, mClsLabels);
    }

    if (mIsCalib) {
      std::vector<CalibInfoCluster>* clusterCalInfo = mClusterer.getInfoFromCluster();
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOCALCLUS", 0, Lifetime::Timeframe}, *clusterCalInfo);
    }

    if (mIsCosmic) {
      std::vector<CosmicInfo>* cosmicInfo = mCosmicProcessor.getCosmicInfo();
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOCOSMICS", 0, Lifetime::Timeframe}, *cosmicInfo);
      std::vector<CalibInfoTrackCl>* cosmicTrack = mCosmicProcessor.getCosmicTrack();
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOTRACKCOS", 0, Lifetime::Timeframe}, *cosmicTrack);
      std::vector<int>* cosmicTrackSize = mCosmicProcessor.getCosmicTrackSize();
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOTRACKSIZE", 0, Lifetime::Timeframe}, *cosmicTrackSize);
    }

    mTimer.Stop();
  }

  void endOfStream(EndOfStreamContext& ec)
  {
    LOGF(INFO, "TOF Clusterer total timing: Cpu: %.3e Real: %.3e s in %d slots",
         mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }

 private:
  DigitDataReader mReader; ///< Digit reader
  Clusterer mClusterer;    ///< Cluster finder
  CosmicProcessor mCosmicProcessor; ///< Cosmics finder
  TStopwatch mTimer;

  std::vector<Cluster> mClustersArray; ///< Array of clusters
  MCLabelContainer mClsLabels;
  std::vector<CalibInfoCluster> mClusterCalInfo; ///< Array of clusters
};

o2::framework::DataProcessorSpec getTOFClusterizerSpec(bool useMC, bool useCCDB, bool doCalib, bool isCosmic)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofdigits", o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("readoutwin", o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  if (useCCDB) {
    inputs.emplace_back("tofccdbLHCphase", o2::header::gDataOriginTOF, "LHCphase");
    inputs.emplace_back("tofccdbChannelCalib", o2::header::gDataOriginTOF, "ChannelCalib");
  }
  if (useMC) {
    inputs.emplace_back("tofdigitlabels", o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }

  if (doCalib) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "INFOCALCLUS", 0, Lifetime::Timeframe);
  }
  if (isCosmic) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "INFOCOSMICS", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "INFOTRACKCOS", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "INFOTRACKSIZE", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TOFClusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLClustererTask>(useMC, useCCDB, doCalib, isCosmic)},
    Options{{"cluster-time-window", VariantType::Int, 5000, {"time window for clusterization in ps"}}}};
}

} // end namespace tof
} // end namespace o2
