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
#include "TOFReconstruction/DataReader.h"
#include "DataFormatsTOF/Cluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "TOFCalibration/CalibTOFapi.h"
#include "TStopwatch.h"

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

 public:
  explicit TOFDPLClustererTask(bool useMC, bool useCCDB) : mUseMC(useMC), mUseCCDB(useCCDB) {}
  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
    mTimer.Stop();
    mTimer.Reset();
  }

  void run(framework::ProcessingContext& pc)
  {
    mTimer.Start(false);
    // get digit data
    auto digits = pc.inputs().get<gsl::span<o2::tof::Digit>>("tofdigits");
    auto row = pc.inputs().get<std::vector<o2::tof::ReadoutWindowData>*>("readoutwin");

    auto header = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("tofdigits").header);
    mClusterer.setFirstOrbit(header->firstTForbit);

    //RSTODO: below is a hack, to remove once the framework will start propagating the header.firstTForbit
    //Here I extract the orbit/BC from the abs.BC, since the triggerer orbit/bunch are not set. Then why they are needed?
    if (digits.size()) {
      auto bcabs = digits[0].getBC();
      auto ir0 = o2::raw::HBFUtils::Instance().getFirstIRofTF({uint16_t(bcabs % Geo::BC_IN_ORBIT), uint32_t(bcabs / Geo::BC_IN_ORBIT)});
      mClusterer.setFirstOrbit(ir0.orbit);
    }

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

    for (int i = 0; i < row->size(); i++) {
      //printf("# TOF readout window for clusterization = %d/%lu (N digits = %d)\n", i, row->size(), row->at(i).size());
      auto digitsRO = row->at(i).getBunchChannelData(digits);

      mReader.setDigitArray(&digitsRO);
      if (mUseMC) {
        mClusterer.process(mReader, mClustersArray, &(labelvector->at(i)));
      } else
        mClusterer.process(mReader, mClustersArray, nullptr);
    }
    LOG(INFO) << "TOF CLUSTERER : TRANSFORMED " << digits.size()
              << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send clusters
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe}, mClustersArray);
    // send labels
    if (mUseMC)
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, mClsLabels);

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
  TStopwatch mTimer;

  std::vector<Cluster> mClustersArray; ///< Array of clusters
  MCLabelContainer mClsLabels;
};

o2::framework::DataProcessorSpec getTOFClusterizerSpec(bool useMC, bool useCCDB)
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

  return DataProcessorSpec{
    "TOFClusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLClustererTask>(useMC, useCCDB)},
    Options{/* for the moment no options */}};
}

} // end namespace tof
} // end namespace o2
