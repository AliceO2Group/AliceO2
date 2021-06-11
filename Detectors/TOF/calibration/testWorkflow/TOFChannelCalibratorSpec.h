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
#include "DataFormatsTOF/CalibInfoCluster.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include <limits>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

template <class T>
class TOFChannelCalibDevice : public o2::framework::Task
{

  using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
  using LHCphase = o2::dataformats::CalibLHCphaseTOF;

 public:
  explicit TOFChannelCalibDevice(bool useCCDB, bool attachChannelOffsetToLHCphase, bool isCosmics) : mUseCCDB(useCCDB), mAttachToLHCphase(attachChannelOffsetToLHCphase), mCosmics(isCosmics) {}
  void init(o2::framework::InitContext& ic) final
  {

    int minEnt = ic.options().get<int>("min-entries"); //std::max(50, ic.options().get<int>("min-entries"));
    int nb = std::max(500, ic.options().get<int>("nbins"));
    float range = ic.options().get<float>("range");
    int isTest = ic.options().get<bool>("do-TOF-channel-calib-in-test-mode");
    bool updateAtEORonly = ic.options().get<bool>("update-at-end-of-run-only");
    int64_t slotL = ic.options().get<int64_t>("tf-per-slot");
    int64_t delay = ic.options().get<int64_t>("max-delay");
    int updateInterval = ic.options().get<int64_t>("update-interval");
    int deltaUpdateInterval = ic.options().get<int64_t>("delta-update-interval");
    mCalibrator = std::make_unique<o2::tof::TOFChannelCalibrator<T>>(minEnt, nb, range);

    // default behaviour is to have only 1 slot at a time, accepting everything for it till the
    // minimum statistics is reached;
    // if one defines a different slot length and delay,
    // the usual time slot calibration behaviour is used;
    // if one defines that the calibration should happen only at the end of the run,
    // then the slot length and delay won't matter
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setCheckIntervalInfiniteSlot(updateInterval);
    mCalibrator->setCheckDeltaIntervalInfiniteSlot(deltaUpdateInterval);
    mCalibrator->setMaxSlotsDelay(delay);

    if (updateAtEORonly) { // has priority over other settings
      mCalibrator->setUpdateAtTheEndOfRunOnly();
    }

    mCalibrator->setIsTest(isTest);
    mCalibrator->setDoCalibWithCosmics(mCosmics);

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

    long startTimeLHCphase;
    long startTimeChCalib;

    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime; // is this the timestamp of the current TF?

    if (mUseCCDB) { // read calibration objects from ccdb
      LHCphase lhcPhaseObjTmp;
      /*
      // for now this part is not implemented; below, the sketch of how it should be done
      if (mAttachToLHCphase) {
        // if I want to take the LHCclockphase just produced, I need to loop over what the previous spec produces:
        int nSlots = pc.inputs().getNofParts(0);
        assert(pc.inputs().getNofParts(1) == nSlots);

        int lhcphaseIndex = -1;
        for (int isl = 0; isl < nSlots; isl++) {
          const auto wrp = pc.inputs().get<CcdbObjectInfo*>("clbInfo", isl);
          if (wrp->getStartValidityTimestamp() > tfcounter) { // replace tfcounter with the timestamp of the TF
            lhxphaseIndex = isl - 1;
            break;
          }
        }
        if (lhcphaseIndex == -1) {
          // no new object found, use CCDB
         auto lhcPhase = pc.inputs().get<LHCphase*>("tofccdbLHCphase");
          lhcPhaseObjTmp = std::move(*lhcPhase);
        }
        else {
          const auto pld = pc.inputs().get<gsl::span<char>>("clbPayload", lhcphaseIndex); // this is actually an image of TMemFile
          // now i need to make a LHCphase object; Ruben suggested how, I did not try yet
         // ...
        }
      }
      else {
      */
      auto lhcPhase = pc.inputs().get<LHCphase*>("tofccdbLHCphase");
      lhcPhaseObjTmp = std::move(*lhcPhase);
      auto channelCalib = pc.inputs().get<TimeSlewing*>("tofccdbChannelCalib");
      TimeSlewing channelCalibObjTmp = std::move(*channelCalib);

      mPhase = lhcPhaseObjTmp;
      mTimeSlewing = channelCalibObjTmp;

      startTimeLHCphase = pc.inputs().get<long>("startTimeLHCphase");
      startTimeChCalib = pc.inputs().get<long>("startTimeChCalib");
    } else {
      startTimeLHCphase = 0;
      startTimeChCalib = 0;
    }

    LOG(DEBUG) << "startTimeLHCphase = " << startTimeLHCphase << ",  startTimeChCalib = " << startTimeChCalib;

    mcalibTOFapi = new o2::tof::CalibTOFapi(long(0), &mPhase, &mTimeSlewing); // TODO: should we replace long(0) with tfcounter defined at the beginning of the method? we need the timestamp of the TF

    mCalibrator->setCalibTOFapi(mcalibTOFapi);

    if ((tfcounter - startTimeChCalib) > 60480000) { // number of TF in 1 week: 7*24*3600/10e-3 - with TF = 10 ms
      LOG(INFO) << "Enlarging the range of the booked histogram since the latest CCDB entry is too old";
      mCalibrator->setRange(mCalibrator->getRange() * 10); // we enlarge the range for the calibration in case the last valid object is too old (older than 1 week)
    }

    if (!mCosmics) {
      auto data = pc.inputs().get<gsl::span<T>>("input");
      LOG(INFO) << "Processing TF " << tfcounter << " with " << data.size() << " tracks";
      mCalibrator->process(tfcounter, data);
      sendOutput(pc.outputs());
    } else {
      auto data = pc.inputs().get<gsl::span<T>>("input");
      LOG(INFO) << "Processing TF " << tfcounter << " with " << data.size() << " tracks";
      mCalibrator->process(tfcounter, data);
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::tof::TOFChannelCalibrator<T>> mCalibrator;
  o2::tof::CalibTOFapi* mcalibTOFapi = nullptr;
  LHCphase mPhase;
  TimeSlewing mTimeSlewing;
  bool mUseCCDB = false;
  bool mAttachToLHCphase = false; // whether to use or not previously defined LHCphase
  bool mCosmics = false;

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

      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_CHANCALIB", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_CHANCALIB", i}, w);            // root-serialized
    }
    if (payloadVec.size()) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }
};

} // namespace calibration

namespace framework
{

template <class T>
DataProcessorSpec getTOFChannelCalibDeviceSpec(bool useCCDB, bool attachChannelOffsetToLHCphase = false, bool isCosmics = false)
{
  constexpr int64_t INFINITE_TF_int64 = std::numeric_limits<long>::max();
  using device = o2::calibration::TOFChannelCalibDevice<T>;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_CHANCALIB"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_CHANCALIB"});

  std::vector<InputSpec> inputs;
  if (!isCosmics) {
    inputs.emplace_back("input", "TOF", "CALIBDATA");
  } else {
    inputs.emplace_back("input", "TOF", "INFOCALCLUS");
  }

  if (useCCDB) {
    inputs.emplace_back("tofccdbLHCphase", o2::header::gDataOriginTOF, "LHCphase");
    inputs.emplace_back("tofccdbChannelCalib", o2::header::gDataOriginTOF, "ChannelCalib");
    inputs.emplace_back("startTimeLHCphase", o2::header::gDataOriginTOF, "StartLHCphase");
    inputs.emplace_back("startTimeChCalib", o2::header::gDataOriginTOF, "StartChCalib");
  }

  return DataProcessorSpec{
    "calib-tofchannel-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(useCCDB, attachChannelOffsetToLHCphase, isCosmics)},
    Options{
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit channel histos"}},
      {"nbins", VariantType::Int, 1000, {"number of bins for t-texp"}},
      {"range", VariantType::Float, 24000.f, {"range for t-text"}},
      {"do-TOF-channel-calib-in-test-mode", VariantType::Bool, false, {"to run in test mode for simplification"}},
      {"ccdb-path", VariantType::String, "http://ccdb-test.cern.ch:8080", {"Path to CCDB"}},
      {"update-at-end-of-run-only", VariantType::Bool, false, {"to update the CCDB only at the end of the run; has priority over calibrating in time slots"}},
      {"tf-per-slot", VariantType::Int64, INFINITE_TF_int64, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int64, 0ll, {"number of slots in past to consider"}},
      {"update-interval", VariantType::Int64, 10ll, {"number of TF after which to try to finalize calibration"}},
      {"delta-update-interval", VariantType::Int64, 10ll, {"number of TF after which to try to finalize calibration, if previous attempt failed"}}}};
}

} // namespace framework
} // namespace o2

#endif
