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

/// \file TPCMergeIntegrateClusterspec.cxx
/// \brief device for merging the integrated TPC clusters in larger contiguous time intervals
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 30, 2023

#include "TPCWorkflow/TPCMergeIntegrateClusterSpec.h"
#include "TPCWorkflow/TPCIntegrateClusterSpec.h"
#include "DetectorsCalibration/IntegratedClusterCalibrator.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "Framework/DataTakingContext.h"
#include "TPCCalibration/IDCFactorization.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/CalDet.h"

#include <numeric>

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCMergeIntegrateClusters : public Task
{
 public:
  /// \construcor
  TPCMergeIntegrateClusters(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mCalibrator = std::make_unique<o2::calibration::IntegratedClusterCalibrator<ITPCC>>();
    mEnableWritingPadStatusMap = ic.options().get<bool>("enableWritingPadStatusMap");
    const auto slotLength = ic.options().get<uint32_t>("tf-per-slot");
    const auto maxDelay = ic.options().get<uint32_t>("max-delay");
    const auto debug = ic.options().get<bool>("debug");
    mNthreads = ic.options().get<int>("nthreads");
    o2::tpc::IDCFactorization::setNThreads(mNthreads);
    mProcess3D = ic.options().get<bool>("process-3D-currents");
    mCalibrator->setSlotLength(slotLength);
    mCalibrator->setMaxSlotsDelay(maxDelay);
    // in case of 3D output set special debug output
    if (debug && mProcess3D) {
      mDump3D = true;
      mCalibrator->setDebug(false);
    } else {
      mCalibrator->setDebug(debug);
    }

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

    const auto currents = pc.inputs().get<ITPCC*>(pc.inputs().get("itpcc"));

    // accumulate the currents
    mCalibrator->process(mCalibrator->getCurrentTFInfo().tfCounter, *currents);

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

  static constexpr header::DataDescription getDataDescriptionCCDB() { return header::DataDescription{"ITPCC"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBITPCPadFlag() { return header::DataDescription{"ITPCCalibFlags"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBITPC0() { return header::DataDescription{"ITPC0Calib"}; }

 private:
  std::unique_ptr<o2::calibration::IntegratedClusterCalibrator<ITPCC>> mCalibrator; ///< calibrator object for creating the pad-by-pad gain map
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                           ///< info for CCDB request
  std::string mMetaFileDir{};                                                       ///< output dir for meta data
  std::string mCalibFileDir{};                                                      ///< output dir for calib objects
  o2::framework::DataTakingContext mDataTakingContext{};                            ///< run information for meta data
  bool mSetDataTakingCont{true};                                                    ///< flag for setting data taking context only once
  bool mDumpCalibData{false};                                                       ///< dump the calibration data as a calibration file
  bool mProcess3D{false};                                                           ///< flag if the 3D TPC currents are expected as input
  bool mDump3D{false};                                                              ///< flag if 3D debug object will be dumped
  int mNthreads{1};                                                                 ///< number of threads used for the factorization
  std::unique_ptr<CalDet<PadFlags>> mPadFlagsMap;                                   ///< status flag for each pad (i.e. if the pad is dead). This map is buffered to check if something changed, when a new map is created
  bool mEnableWritingPadStatusMap{false};                                           ///< do not store the pad status map in the CCDB

  void sendOutput(DataAllocator& output)
  {
    auto calibrations = std::move(*mCalibrator).getCalibs();
    const auto& intervals = mCalibrator->getTimeIntervals();
    assert(calibrations.size() == intervals.size());
    for (unsigned int i = 0; i < calibrations.size(); i++) {
      auto& object = calibrations[i];
      o2::ccdb::CcdbObjectInfo info(CDBTypeMap.at(CDBType::CalITPC1), std::string{}, std::string{}, std::map<std::string, std::string>{}, intervals[i].first, intervals[i].second);

      // perform factorization in case of 3D currents used as input
      std::unique_ptr<std::vector<char>> imageFlagMap;
      std::unique_ptr<std::vector<char>> imageITPC0;
      if (mProcess3D) {
        std::vector<uint32_t> crus(o2::tpc::CRU::MaxCRU);
        std::iota(crus.begin(), crus.end(), 0);
        const unsigned int nIntegrationIntervals = object.getEntries() / Mapper::getNumberOfPadsPerSide();
        IDCFactorization factorizeqMax(nIntegrationIntervals, 1, crus);
        IDCFactorization factorizeqTot(nIntegrationIntervals, 1, crus);
        IDCFactorization factorizeNCl(nIntegrationIntervals, 1, crus);
        LOGP(info, "Processing {} integration intervals", nIntegrationIntervals);

        for (int cru = 0; cru < o2::tpc::CRU::MaxCRU; ++cru) {
          CRU cruTmp(cru);
          const Side side = cruTmp.side();
          const unsigned int region = cruTmp.region();
          const unsigned int sector = cruTmp.sector();
          for (int interval = 0; interval < nIntegrationIntervals; ++interval) {
            const unsigned int indexStart = interval * Mapper::getNumberOfPadsPerSide() + (sector % SECTORSPERSIDE) * Mapper::getPadsInSector() + Mapper::GLOBALPADOFFSET[region];
            const unsigned int indexEnd = indexStart + Mapper::PADSPERREGION[region];
            const auto& currqMax = (side == Side::A) ? object.mIQMaxA : object.mIQMaxC;
            const auto& currqTot = (side == Side::A) ? object.mIQTotA : object.mIQTotC;
            const auto& currNCl = (side == Side::A) ? object.mINClA : object.mINClC;

            // check if values are empty -> dummy input
            if (std::all_of(currNCl.begin() + indexStart, currNCl.begin() + indexEnd, [](float x) { return x == 0; })) {
              continue;
            }

            // copy currents for factorization (ToDo: optimize the class for factorization such that no copy is required)
            factorizeqMax.setIDCs(std::vector<float>(currqMax.begin() + indexStart, currqMax.begin() + indexEnd), cru, interval);
            factorizeqTot.setIDCs(std::vector<float>(currqTot.begin() + indexStart, currqTot.begin() + indexEnd), cru, interval);
            factorizeNCl.setIDCs(std::vector<float>(currNCl.begin() + indexStart, currNCl.begin() + indexEnd), cru, interval);
          }
        }

        if (mDump3D) {
          LOGP(info, "Writing debug output to file");
          factorizeqMax.setTimeStamp(info.getStartValidityTimestamp());
          factorizeqTot.setTimeStamp(info.getStartValidityTimestamp());
          factorizeNCl.setTimeStamp(info.getStartValidityTimestamp());
          const int run = std::stoi(mDataTakingContext.runNumber);
          factorizeqMax.setRun(run);
          factorizeqTot.setRun(run);
          factorizeNCl.setRun(run);
          factorizeqMax.dumpLargeObjectToFile(fmt::format("IDCFactorization_qMax_{}_{}.root", info.getStartValidityTimestamp(), info.getEndValidityTimestamp()).data());
          factorizeqTot.dumpLargeObjectToFile(fmt::format("IDCFactorization_qTot_{}_{}.root", info.getStartValidityTimestamp(), info.getEndValidityTimestamp()).data());
          factorizeNCl.dumpLargeObjectToFile(fmt::format("IDCFactorization_NCl_{}_{}.root", info.getStartValidityTimestamp(), info.getEndValidityTimestamp()).data());
        }

        // perform factorization (I0,I1,outlier map)
        factorizeqMax.factorizeIDCs(true, false); // normalize to pad size
        factorizeqTot.factorizeIDCs(true, false); // normalize to pad size
        factorizeNCl.factorizeIDCs(false, false); // do not normalize to pad size

        // copy calibration data
        object.mIQMaxA = factorizeqMax.getIDCOneVec(Side::A);
        object.mIQMaxC = factorizeqMax.getIDCOneVec(Side::C);
        object.mIQTotA = factorizeqTot.getIDCOneVec(Side::A);
        object.mIQTotC = factorizeqTot.getIDCOneVec(Side::C);
        object.mINClA = factorizeNCl.getIDCOneVec(Side::A);
        object.mINClC = factorizeNCl.getIDCOneVec(Side::C);

        // storing pad status map in CCDB
        auto padStatusMap = factorizeNCl.getPadStatusMap();
        if (mEnableWritingPadStatusMap) {
          // always store the first map
          if (!mPadFlagsMap) {
            mPadFlagsMap = std::move(padStatusMap);
            o2::ccdb::CcdbObjectInfo ccdbInfoPadFlags(CDBTypeMap.at(CDBType::CalIDCPadStatusMapA), std::string{}, std::string{}, std::map<std::string, std::string>{}, intervals[i].first, intervals[i].second);
            imageFlagMap = o2::ccdb::CcdbApi::createObjectImage(mPadFlagsMap.get(), &ccdbInfoPadFlags);
            LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoPadFlags.getPath(), ccdbInfoPadFlags.getFileName(), imageFlagMap->size(), ccdbInfoPadFlags.getStartValidityTimestamp(), ccdbInfoPadFlags.getEndValidityTimestamp());
            output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBITPCPadFlag(), 0}, *imageFlagMap.get());
            output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBITPCPadFlag(), 0}, ccdbInfoPadFlags);
          } else {
            // check if map changed. if it changed update the map in the CCDB and store new map in buffer
            if (!(*padStatusMap.get() == *mPadFlagsMap.get())) {
              LOGP(info, "Pad status map changed");
              o2::ccdb::CcdbObjectInfo ccdbInfoPadFlags(CDBTypeMap.at(CDBType::CalIDCPadStatusMapA), std::string{}, std::string{}, std::map<std::string, std::string>{}, intervals[i].first, intervals[i].second);
              imageFlagMap = o2::ccdb::CcdbApi::createObjectImage(mPadFlagsMap.get(), &ccdbInfoPadFlags);
              LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoPadFlags.getPath(), ccdbInfoPadFlags.getFileName(), imageFlagMap->size(), ccdbInfoPadFlags.getStartValidityTimestamp(), ccdbInfoPadFlags.getEndValidityTimestamp());
              output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBITPCPadFlag(), 0}, *imageFlagMap.get());
              output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBITPCPadFlag(), 0}, ccdbInfoPadFlags);
              mPadFlagsMap = std::move(padStatusMap);
            }
          }
        }

        // moving I0 to calibration object
        ITPCZero itpczero;
        itpczero.mIQMaxA = std::move(factorizeqMax).getIDCZero(Side::A);
        itpczero.mIQMaxC = std::move(factorizeqMax).getIDCZero(Side::C);
        itpczero.mIQTotA = std::move(factorizeqTot).getIDCZero(Side::A);
        itpczero.mIQTotC = std::move(factorizeqTot).getIDCZero(Side::C);
        itpczero.mINClA = std::move(factorizeNCl).getIDCZero(Side::A);
        itpczero.mINClC = std::move(factorizeNCl).getIDCZero(Side::C);

        o2::ccdb::CcdbObjectInfo ccdbInfoITPC0(CDBTypeMap.at(CDBType::CalITPC0), std::string{}, std::string{}, std::map<std::string, std::string>{}, intervals[i].first, intervals[i].second);
        imageITPC0 = o2::ccdb::CcdbApi::createObjectImage(&itpczero, &ccdbInfoITPC0);
        LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoITPC0.getPath(), ccdbInfoITPC0.getFileName(), imageITPC0->size(), ccdbInfoITPC0.getStartValidityTimestamp(), ccdbInfoITPC0.getEndValidityTimestamp());
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBITPC0(), 0}, *imageITPC0.get());
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBITPC0(), 0}, ccdbInfoITPC0);
      }

      auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &info);
      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", info.getPath(), info.getFileName(), image->size(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDB(), i}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDB(), i}, info);

      if (mDumpCalibData && mCalibFileDir != "/dev/null") {
        std::string calibFName = fmt::format("itpc_cal_data_{}_{}.root", info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
        try {
          std::ofstream calFile(fmt::format("{}{}", mCalibFileDir, calibFName), std::ios::out | std::ios::binary);
          calFile.write(image->data(), image->size());
          calFile.close();
          if (imageFlagMap) {
            std::string calibFNameMap = fmt::format("itpc_cal_map_data_{}_{}.root", info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
            std::ofstream calFileMap(fmt::format("{}{}", mCalibFileDir, calibFNameMap), std::ios::out | std::ios::binary);
            calFileMap.write(imageFlagMap->data(), imageFlagMap->size());
            calFileMap.close();
          }
          if (imageITPC0) {
            std::string calibFNameI0 = fmt::format("itpc_cal0_data_{}_{}.root", info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
            std::ofstream calFileI0(fmt::format("{}{}", mCalibFileDir, calibFNameI0), std::ios::out | std::ios::binary);
            calFileI0.write(imageITPC0->data(), imageITPC0->size());
            calFileI0.close();
          }
        } catch (std::exception const& e) {
          LOG(error) << "Failed to store ITPCC calibration data file " << calibFName << ", reason: " << e.what();
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

o2::framework::DataProcessorSpec getTPCMergeIntegrateClusterSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("itpcc", o2::header::gDataOriginTPC, getDataDescriptionTPCC(), 0, Lifetime::Sporadic);
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                false,                          // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCMergeIntegrateClusters::getDataDescriptionCCDB()}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCMergeIntegrateClusters::getDataDescriptionCCDB()}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCMergeIntegrateClusters::getDataDescriptionCCDBITPCPadFlag()}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCMergeIntegrateClusters::getDataDescriptionCCDBITPCPadFlag()}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCMergeIntegrateClusters::getDataDescriptionCCDBITPC0()}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCMergeIntegrateClusters::getDataDescriptionCCDBITPC0()}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-merge-integrated-clusters",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCMergeIntegrateClusters>(ccdbRequest)},
    Options{
      {"debug", VariantType::Bool, false, {"Write debug output files"}},
      {"tf-per-slot", VariantType::UInt32, 1000u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 3u, {"number of slots in past to consider"}},
      {"output-dir", VariantType::String, "none", {"calibration files output directory, must exist"}},
      {"meta-output-dir", VariantType::String, "/dev/null", {"calibration metadata output directory, must exist (if not /dev/null)"}},
      {"dump-calib-data", VariantType::Bool, false, {"Dump ITPCC calibration data to file"}},
      {"process-3D-currents", VariantType::Bool, false, {"Process full 3D currents instead of 1D integrated only currents"}},
      {"enableWritingPadStatusMap", VariantType::Bool, false, {"Write the pad status map to CCDB"}},
      {"nthreads", VariantType::Int, 1, {"Number of threads used for factorization"}}}};
}

} // end namespace tpc
} // end namespace o2
