// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_TOFCHANNEL_CALIBRATOR_H
#define O2_CALIBRATION_TOFCHANNEL_CALIBRATOR_H

/// @file   TOFChannelCalibratorSpec.h
/// @brief  Device to calibrate TOF channles (offsets)

#include "TOFCalibration/TOFChannelCalibrator.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class TOFChannelCalibDevice : public o2::framework::Task
{

 using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
 using LHCphase = o2::dataformats::CalibLHCphaseTOF;
 public:
  explicit TOFChannelCalibDevice(bool useCCDB) : mUseCCDB(useCCDB) {}
  void init(o2::framework::InitContext& ic) final
  {

    int minEnt = std::max(50, ic.options().get<int>("min-entries"));
    int nb = std::max(500, ic.options().get<int>("nbins"));
    float range = ic.options().get<float>("range");
    int isTest = ic.options().get<bool>("do-TOF-channel-calib-in-test-mode");
    //int slotL = ic.options().get<int>("tf-per-slot");
    //int delay = ic.options().get<int>("max-delay");
    mCalibrator = std::make_unique<o2::tof::TOFChannelCalibrator>(minEnt, nb, range);
    mCalibrator->setUpdateAtTheEndOfRunOnly();
    mCalibrator->isTest(isTest);
    //    mCalibrator->setSlotLength(slotL); // to be done: we need to configure this so that it just does the calibration at the end of the run
    // mCalibrator->setMaxSlotsDelay(delay); // to be done: we need to configure this so that it just does the calibration at the end of the run

    // calibration objects set to zero
    mPhase.addLHCphase(0, 0);
    mPhase.addLHCphase(2000000000, 0);
	
    for (int ich = 0; ich < TimeSlewing::NCHANNELS; ich++) {
      mTimeSlewing.addTimeSlewingInfo(ich, 0, 0);
      int sector = ich / TimeSlewing::NCHANNELXSECTOR;
      int channelInSector = ich % TimeSlewing::NCHANNELXSECTOR;
      mTimeSlewing.setFractionUnderPeak(sector, channelInSector, 1);
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {

    //LHCphase lhcPhaseObj;
    //TimeSlewingParam channelCalibObj;
    long startTimeLHCphase;
    long startTimeChCalib;
    
    if (mUseCCDB) { // read calibration objects from ccdb
      // check LHC phase
      auto lhcPhase = pc.inputs().get<LHCphase*>("tofccdbLHCphase");
      auto channelCalib = pc.inputs().get<TimeSlewing*>("tofccdbChannelCalib");
      
      LHCphase lhcPhaseObjTmp = std::move(*lhcPhase);
      TimeSlewing channelCalibObjTmp = std::move(*channelCalib);
      
      // make a copy in global scope
      //lhcPhaseObj = lhcPhaseObjTmp;
      //channelCalibObj = channelCalibObjTmp;
      mPhase = lhcPhaseObjTmp;
      mTimeSlewing = channelCalibObjTmp;
      
      startTimeLHCphase = pc.inputs().get<long>("startTimeLHCphase");
      startTimeChCalib = pc.inputs().get<long>("startTimeChCalib");
    }
    else {
      startTimeLHCphase = 0;
      startTimeChCalib = 0;
    }

    for (int ich = 0; ich < 10; ich++){
      LOG(INFO) << "for test, checking channel " << ich << ": offset = " << mTimeSlewing.evalTimeSlewing(ich, 0);
    }
    mcalibTOFapi = new o2::tof::CalibTOFapi(long(0), &mPhase, &mTimeSlewing);  // TODO: should we replace long(0) with tfcounter defined below? we need the timestamp of the TF
    
    mCalibrator->setCalibTOFapi(mcalibTOFapi);
    
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    if ((tfcounter - startTimeChCalib) > 60480000) { // number of TF in 1 week: 7*24*3600/10e-3 - with TF = 10 ms
      mCalibrator->setRange(mCalibrator->getRange()*10); // we enlarge the range for the calibration in case the last valid object is too old (older than 1 week)
    }
  
    auto data = pc.inputs().get<gsl::span<o2::dataformats::CalibInfoTOF>>("input");
    LOG(INFO) << "Processing TF " << tfcounter << " with " << data.size() << " tracks";
    mCalibrator->process(tfcounter, data);
    //sendOutput(pc.outputs());
    const auto& infoVec = mCalibrator->getTimeSlewingInfoVector();
    LOG(INFO) << "Created " << infoVec.size() << " objects for TF " << tfcounter;
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(INFO) << "Finalizing calibration";
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::tof::TOFChannelCalibrator> mCalibrator;
  o2::tof::CalibTOFapi* mcalibTOFapi = nullptr;
  LHCphase mPhase;
  TimeSlewing mTimeSlewing;
  bool mUseCCDB = false;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
    using clbUtils = o2::calibration::Utils;
    const auto& payloadVec = mCalibrator->getTimeSlewingVector();
    auto& infoVec = mCalibrator->getTimeSlewingInfoVector(); // use non-const version as we update it
    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, i}, *image.get()); // vector<char>
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, i}, w);               // root-serialized
    }
    if (payloadVec.size()) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTOFChannelCalibDeviceSpec(bool useCCDB)
{
  using device = o2::calibration::TOFChannelCalibDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});

  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "DUM", "CALIBDATA");
  if (useCCDB) {
    inputs.emplace_back("tofccdbLHCphase", o2::header::gDataOriginTOF, "LHCphase");
    inputs.emplace_back("tofccdbChannelCalib", o2::header::gDataOriginTOF, "ChannelCalib");
    inputs.emplace_back("startTimeLHCphase", o2::header::gDataOriginTOF, "StartLHCphase");
    inputs.emplace_back("startTimeChCalib", o2::header::gDataOriginTOF, "StartChCalib");
  }

  //  inputs.emplace_back("testFlag", "DUM", "TESTFLAG");
		      
  return DataProcessorSpec{
    "calib-tofchannel-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(useCCDB)},
    Options{
      //      {"tf-per-slot", VariantType::Int, 5, {"number of TFs per calibration time slot"}},
      //{"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit channel histos"}},
      {"nbins", VariantType::Int, 1000, {"number of bins for t-texp"}},
      {"range", VariantType::Float, 24000.f, {"range for t-text"}},
      {"do-TOF-channel-calib-in-test-mode", VariantType::Bool, false, {"to run in test mode for simplification"}},
      {"ccdb-path", VariantType::String, "http://ccdb-test.cern.ch:8080", {"Path to CCDB"}}}};
  
}

} // namespace framework
} // namespace o2

#endif
