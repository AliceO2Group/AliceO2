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

/// \file TOFMergeIntegrateClusterSpec.cxx
/// \brief device for merging the integrated TOF clusters in larger contiguous time intervals
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 21, 2023

#include "TOFWorkflowUtils/TOFMergeIntegrateClusterSpec.h"
#include "DetectorsCalibration/IntegratedClusterCalibrator.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "Framework/DataTakingContext.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class TOFMergeIntegrateClusters : public Task
{
 public:
  /// \construcor
  TOFMergeIntegrateClusters(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mCalibrator = std::make_unique<o2::calibration::IntegratedClusterCalibrator<ITOFC>>();
    const auto slotLength = ic.options().get<uint32_t>("tf-per-slot");
    const auto maxDelay = ic.options().get<uint32_t>("max-delay");
    const auto debug = ic.options().get<bool>("debug");
    mCalibrator->setSlotLength(slotLength);
    mCalibrator->setMaxSlotsDelay(maxDelay);
    mCalibrator->setDebug(debug);
    mCalibFileDir = ic.options().get<std::string>("output-dir");
    if (mCalibFileDir != "/dev/null") {
      mCalibFileDir = o2::utils::Str::rectifyDirectory(mCalibFileDir);
    }
    mMetaFileDir = ic.options().get<std::string>("meta-output-dir");
    if (mMetaFileDir != "/dev/null") {
      mMetaFileDir = o2::utils::Str::rectifyDirectory(mMetaFileDir);
    }
    mDumpCalibData = ic.options().get<bool>("dump-calib-data");
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());

    // set data taking context only once
    if (mSetDataTakingCont) {
      mDataTakingContext = pc.services().get<DataTakingContext>();
      mSetDataTakingCont = false;
    }

    ITOFC tofcurr;
    tofcurr.mITOFCNCl = pc.inputs().get<std::vector<float>>(pc.inputs().get("itofcn"));
    tofcurr.mITOFCQ = pc.inputs().get<std::vector<float>>(pc.inputs().get("itofcq"));

    // accumulate the currents
    mCalibrator->process(mCalibrator->getCurrentTFInfo().tfCounter, tofcurr);

    LOGP(debug, "Created {} objects for TF {} and time stamp {}", mCalibrator->getTFinterval().size(), mCalibrator->getCurrentTFInfo().tfCounter, mCalibrator->getCurrentTFInfo().creation);

    if (mCalibrator->hasCalibrationData()) {
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration. Dumping all objects");
    // just write everything out
    for (int i = 0; i < mCalibrator->getNSlots(); ++i) {
      mCalibrator->finalizeSlot(mCalibrator->getSlot(i));
    }
    sendOutput(eos.outputs());
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final { o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj); }

  static constexpr header::DataDescription getDataDescriptionCCDB() { return header::DataDescription{"ITOFC"}; }

 private:
  std::unique_ptr<o2::calibration::IntegratedClusterCalibrator<ITOFC>> mCalibrator; ///< calibrator object for creating the pad-by-pad gain map
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                           ///< info for CCDB request
  std::string mMetaFileDir{};                                                       ///< output dir for meta data
  std::string mCalibFileDir{};                                                      ///< output dir for calib objects
  o2::framework::DataTakingContext mDataTakingContext{};                            ///< run information for meta data
  bool mSetDataTakingCont{true};                                                    ///< flag for setting data taking context only once
  bool mDumpCalibData{false};                                                       ///< dump the ITOFC as a calibration file

  void sendOutput(DataAllocator& output)
  {
    auto calibrations = std::move(*mCalibrator).getCalibs();
    const auto& intervals = mCalibrator->getTimeIntervals();
    assert(calibrations.size() == intervals.size());
    for (unsigned int i = 0; i < calibrations.size(); i++) {
      const auto& object = calibrations[i];
      o2::ccdb::CcdbObjectInfo info("TOF/Calib/ITOFC", std::string{}, std::string{}, std::map<std::string, std::string>{}, intervals[i].first, intervals[i].second);
      auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &info);
      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", info.getPath(), info.getFileName(), image->size(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDB(), i}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDB(), i}, info);

      if (mDumpCalibData && mCalibFileDir != "/dev/null") {
        std::string calibFName = fmt::format("itofc_cal_data_{}_{}.root", info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
        try {
          std::ofstream calFile(fmt::format("{}{}", mCalibFileDir, calibFName), std::ios::out | std::ios::binary);
          calFile.write(image->data(), image->size());
          calFile.close();
        } catch (std::exception const& e) {
          LOG(error) << "Failed to store ITOFC calibration data file " << calibFName << ", reason: " << e.what();
        }

        if (mMetaFileDir != "/dev/null") {
          o2::dataformats::FileMetaData calMetaData;
          calMetaData.fillFileData(calibFName);
          calMetaData.setDataTakingContext(mDataTakingContext);
          calMetaData.type = "calib";
          calMetaData.priority = "low";
          auto metaFileNameTmp = fmt::format("{}{}.tmp", mMetaFileDir, calibFName);
          auto metaFileName = fmt::format("{}{}.done", mMetaFileDir, calibFName);
          try {
            std::ofstream metaFileOut(metaFileNameTmp);
            metaFileOut << calMetaData;
            metaFileOut.close();
            std::filesystem::rename(metaFileNameTmp, metaFileName);
          } catch (std::exception const& e) {
            LOG(error) << "Failed to store CTF meta data file " << metaFileName << ", reason: " << e.what();
          }
        }
      }
    }
    mCalibrator->initOutput(); // empty the outputs after they are send
  }
};

o2::framework::DataProcessorSpec getTOFMergeIntegrateClusterSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("itofcn", o2::header::gDataOriginTOF, "ITOFCN", 0, Lifetime::Sporadic);
  inputs.emplace_back("itofcq", o2::header::gDataOriginTOF, "ITOFCQ", 0, Lifetime::Sporadic);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                false,                          // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TOFMergeIntegrateClusters::getDataDescriptionCCDB()}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TOFMergeIntegrateClusters::getDataDescriptionCCDB()}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tof-merge-integrated-clusters",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFMergeIntegrateClusters>(ccdbRequest)},
    Options{
      {"debug", VariantType::Bool, false, {"Write debug output files"}},
      {"tf-per-slot", VariantType::UInt32, 1000u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 3u, {"number of slots in past to consider"}},
      {"output-dir", VariantType::String, "none", {"calibration files output directory, must exist"}},
      {"meta-output-dir", VariantType::String, "/dev/null", {"calibration metadata output directory, must exist (if not /dev/null)"}},
      {"dump-calib-data", VariantType::Bool, false, {"Dump ITOFC calibration data to file"}}}};
}

} // end namespace tof
} // end namespace o2
