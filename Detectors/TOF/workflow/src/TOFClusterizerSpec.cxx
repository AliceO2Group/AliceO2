// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include "TOFBase/CalibTOFapi.h"
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRefUtils.h"
#include "TOFBase/Utils.h"
#include "Steer/MCKinematicsReader.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TSystem.h"

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
  bool mUpdateCCDB = false;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::string mCCDBurl;
  o2::tof::CalibTOFapi* mCalibApi = nullptr;

 public:
  explicit TOFDPLClustererTask(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, bool useCCDB, bool doCalib, bool isCosmic, std::string ccdb_url, bool isForCalib) : mGGCCDBRequest(gr), mUseMC(useMC), mUseCCDB(useCCDB), mIsCalib(doCalib), mIsCosmic(isCosmic), mCCDBurl(ccdb_url), mForCalib(isForCalib) {}

  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
    mTimer.Stop();
    mTimer.Reset();
    o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
    mTimeWin = ic.options().get<int>("cluster-time-window");
    LOG(debug) << "Is calibration from cluster on? " << mIsCalib;
    LOG(debug) << "DeltaTime for clusterization = " << mTimeWin << " ps";
    LOG(debug) << "Is cosmics? " << mIsCosmic;

    mClusterer.setCalibFromCluster(mIsCalib);
    mClusterer.setDeltaTforClustering(mTimeWin);
    mClusterer.setCalibStored(mForCalib);

    mMultPerLongBC.resize(o2::base::GRPGeomHelper::instance().getNHBFPerTF() * o2::constants::lhc::LHCMaxBunches);
    std::fill(mMultPerLongBC.begin(), mMultPerLongBC.end(), 0);
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher matcher, void* obj)
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      return;
    }
    if (matcher == ConcreteDataMatcher("TOF", "DiagnosticCal", 0)) {
      mUpdateCCDB = true;
      return;
    }
    if (matcher == ConcreteDataMatcher("TOF", "LHCphaseCal", 0)) {
      mUpdateCCDB = true;
      return;
    }
    if (matcher == ConcreteDataMatcher("TOF", "ChannelCalibCal", 0)) {
      mUpdateCCDB = true;
      return;
    }
  }

  void run(framework::ProcessingContext& pc)
  {
    mTimer.Start(false);
    // reset multiplicity counters
    std::fill(mMultPerLongBC.begin(), mMultPerLongBC.end(), 0);

    // get digit data
    auto digits = pc.inputs().get<gsl::span<o2::tof::Digit>>("tofdigits");
    auto row = pc.inputs().get<gsl::span<o2::tof::ReadoutWindowData>>("readoutwin");
    auto dia = pc.inputs().get<o2::tof::Diagnostic*>("diafreq");
    auto patterns = pc.inputs().get<pmr::vector<unsigned char>>("patterns");
    updateTimeDependentParams(pc);
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    mClusterer.setFirstOrbit(tinfo.firstTForbit);

    auto labelvector = std::make_shared<std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>>();
    if (mUseMC) {
      auto digitlabels = pc.inputs().get<std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>*>("tofdigitlabels");
      *labelvector.get() = std::move(*digitlabels);
      mClusterer.setMCTruthContainer(&mClsLabels);
      mClsLabels.clear();
    }

    if (!mUseCCDB && !mCalibApi) { // calibration objects set to zero
      auto* lhcPhaseDummy = new o2::dataformats::CalibLHCphaseTOF();
      auto* channelCalibDummy = new o2::dataformats::CalibTimeSlewingParamTOF();

      lhcPhaseDummy->addLHCphase(0, 0);
      lhcPhaseDummy->addLHCphase(2000000000, 0);

      for (int ich = 0; ich < o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELS; ich++) {
        channelCalibDummy->addTimeSlewingInfo(ich, 0, 0);
        int sector = ich / o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
        int channelInSector = ich % o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
        channelCalibDummy->setFractionUnderPeak(sector, channelInSector, 1);
      }
      mCalibApi = new o2::tof::CalibTOFapi(long(0), lhcPhaseDummy, channelCalibDummy);
    }

    mCalibApi->setTimeStamp(tinfo.creation / 1000);

    mClusterer.setCalibApi(mCalibApi);

    mClusterer.clearDiagnostic();
    mClusterer.addDiagnostic(*dia);

    // call actual clustering routine
    mClustersArray.clear();
    if (mIsCalib) {
      mClusterer.getInfoFromCluster()->clear();
    }

    if (mIsCosmic) {
      mCosmicProcessor.clear();
    }

    for (unsigned int i = 0; i < row.size(); i++) {
      //fill trm pattern but process them in clusterize since they are for readout windows
      mCalibApi->resetTRMErrors();

      // loop over crates
      int kw = 0;
      for (int crate = 0; crate < 72; crate++) {
        int slot = -1;
        int nwords = kw + row[i].getDiagnosticInCrate(crate);
        int eword;
        for (; kw < nwords; kw++) {
          if (patterns[kw] > 28) { // new slot
            if (slot > -1) {       // fill previous
              mCalibApi->processError(crate, slot, eword);
            }
            slot = patterns[kw] - 28;
            eword = 0;

            if (slot < 3 || slot > 12) { // not a valid slot -> don't fill otherwise mapping fails (slot 1 crate 0 is a bad condition)
              slot = -1;
              //              LOG(info) << "not a valid slot in diagnostic words: slot  = " << slot << " for crate " << crate;
            }
          } else if (slot > -1) { // process error in this slot
            eword += 1 << patterns[kw];
          }
        }
      } // end crate loop

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
    LOG(debug) << "TOF CLUSTERER : TRANSFORMED " << digits.size()
               << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // fill multiplicty counters
    int det[5];
    float pos[3];
    for (const auto& cls : mClustersArray) {
      double timeCl = cls.getTime();
      int channel = cls.getMainContributingChannel();                                          // channel index
      o2::tof::Geo::getVolumeIndices(channel, det);                                            // detector index
      o2::tof::Geo::getPos(det, pos);                                                          // detector position
      timeCl -= sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]) * 33.3564095 - 3000; // subtract minimal arrival time (keeping 3 ns of margin)

      int longBC = int(timeCl * o2::tof::Geo::BC_TIME_INPS_INV);
      if (longBC < 0 || longBC >= mMultPerLongBC.size()) {
        continue;
      }
      mMultPerLongBC[longBC]++;
    }

    // send clusters
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe}, mClustersArray);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERSMULT", 0, Lifetime::Timeframe}, mMultPerLongBC);
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
    LOGF(debug, "TOF Clusterer total timing: Cpu: %.3e Real: %.3e s in %d slots",
         mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }

 private:
  void updateTimeDependentParams(ProcessingContext& pc)
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    static bool initOnceDone = false;
    if (!initOnceDone) { // this params need to be queried only once
      initOnceDone = true;
      const auto bcs = o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling().getFilledBCs();
      for (auto bc : bcs) {
        o2::tof::Utils::addInteractionBC(bc, true);
      }
    }

    if (mUseCCDB) { // read calibration objects from ccdb
      // check LHC phase
      const auto lhcPhaseIn = pc.inputs().get<o2::dataformats::CalibLHCphaseTOF*>("tofccdbLHCphase");
      const auto channelCalibIn = pc.inputs().get<o2::dataformats::CalibTimeSlewingParamTOF*>("tofccdbChannelCalib");
      const auto diagnosticIn = pc.inputs().get<o2::tof::Diagnostic*>("tofccdbDia");

      if (!mCalibApi) {
        o2::dataformats::CalibLHCphaseTOF* lhcPhase = new o2::dataformats::CalibLHCphaseTOF(std::move(*lhcPhaseIn));
        o2::dataformats::CalibTimeSlewingParamTOF* channelCalib = new CalibTimeSlewingParamTOF(std::move(*channelCalibIn));
        o2::tof::Diagnostic* diagnostic = new o2::tof::Diagnostic(std::move(*diagnosticIn));
        mCalibApi = new o2::tof::CalibTOFapi(long(0), lhcPhase, channelCalib, diagnostic);
        mCalibApi->loadDiagnosticFrequencies();
        mUpdateCCDB = false;
      } else { // update if necessary
        if (mUpdateCCDB) {
          LOG(info) << "Update CCDB objects since new";
          delete mCalibApi;
          o2::dataformats::CalibLHCphaseTOF* lhcPhase = new o2::dataformats::CalibLHCphaseTOF(*lhcPhaseIn);
          o2::dataformats::CalibTimeSlewingParamTOF* channelCalib = new CalibTimeSlewingParamTOF(*channelCalibIn);
          o2::tof::Diagnostic* diagnostic = new o2::tof::Diagnostic(std::move(*diagnosticIn));
          mCalibApi = new o2::tof::CalibTOFapi(long(0), lhcPhase, channelCalib, diagnostic);
          mCalibApi->loadDiagnosticFrequencies();
          mUpdateCCDB = false;
        } else {
          // do nothing
        }
      }
    }
  }

  DigitDataReader mReader; ///< Digit reader
  Clusterer mClusterer;    ///< Cluster finder
  CosmicProcessor mCosmicProcessor; ///< Cosmics finder
  TStopwatch mTimer;

  std::vector<Cluster> mClustersArray; ///< Array of clusters
  MCLabelContainer mClsLabels;
  std::vector<CalibInfoCluster> mClusterCalInfo; ///< Array of clusters
  bool mForCalib = false;

  std::vector<int> mMultPerLongBC; ///< MultiplicityPerLongBC
};

o2::framework::DataProcessorSpec getTOFClusterizerSpec(bool useMC, bool useCCDB, bool doCalib, bool isCosmic, std::string ccdb_url, bool isForCalib)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofdigits", o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("readoutwin", o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  inputs.emplace_back("diafreq", o2::header::gDataOriginTOF, "DIAFREQ", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe);

  if (useCCDB) {
    //    inputs.emplace_back("tofccdbLHCphase", "TOF", "StatusTOF", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/Status"));
    inputs.emplace_back("tofccdbDia", "TOF", "DiagnosticCal", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/Diagnostic"));
    inputs.emplace_back("tofccdbLHCphase", "TOF", "LHCphaseCal", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/LHCphase"));
    inputs.emplace_back("tofccdbChannelCalib", "TOF", "ChannelCalibCal", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/ChannelCalib"));
    //    inputs.emplace_back("tofccdbChannelCalib", o2::header::gDataOriginTOF, "ChannelCalib");
  }
  if (useMC) {
    inputs.emplace_back("tofdigitlabels", o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "CLUSTERSMULT", 0, Lifetime::Timeframe);
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
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);
  return DataProcessorSpec{
    "TOFClusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLClustererTask>(ggRequest, useMC, useCCDB, doCalib, isCosmic, ccdb_url, isForCalib)},
    Options{{"cluster-time-window", VariantType::Int, 5000, {"time window for clusterization in ps"}}}};
}

} // end namespace tof
} // end namespace o2
