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

#include "PHOSCalibWorkflow/PHOSEnergyCalibDevice.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ControlService.h"
#include "Framework/CCDBParamSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"

#include <fairmq/Device.h>
#include "FairLogger.h"
#include <string>
#include <filesystem>

using namespace o2::phos;

void PHOSEnergyCalibDevice::init(o2::framework::InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  mOutputDir = ic.options().get<std::string>("output-dir");
  mOutputDir = o2::utils::Str::rectifyDirectory(mOutputDir);
  mMetaFileDir = ic.options().get<std::string>("meta-output-dir");
  mMetaFileDir = o2::utils::Str::rectifyDirectory(mMetaFileDir);
  LOG(error) << "Meta dir=" << mMetaFileDir;

  mPtMin = ic.options().get<float>("ptminmgg");
  mEminHGTime = ic.options().get<float>("eminhgtime");
  mEminLGTime = ic.options().get<float>("eminlgtime");
  mEDigMin = ic.options().get<float>("ecalibdigitmin");
  mECluMin = ic.options().get<float>("ecalibclumin");

  LOG(info) << "Energy calibration options";
  LOG(info) << " output-dif=" << mOutputDir;
  LOG(info) << " meta-output-dir=" << mMetaFileDir;
  LOG(info) << " mgg histo ptMin=" << mPtMin;
  LOG(info) << " Emin for HG time=" << mEminHGTime;
  LOG(info) << " Emin for LG time=" << mEminLGTime;
  LOG(info) << " Emin for out digits=" << mEDigMin;
  LOG(info) << " Cluster Emin for out digits=" << mECluMin;

  mCalibrator = std::make_unique<PHOSEnergyCalibrator>();

  // read calibration and bad map objects and send them to calibrator
  if (!mHasCalib && !mUseCCDB) {
    mCalibParams = std::make_unique<CalibParams>(1); // Create test calibration coefficients
    mCalibrator->setCalibration(mCalibParams.get());
    mBadMap = std::make_unique<BadChannelsMap>(); // Create empty bad map
    mCalibrator->setBadMap(mBadMap.get());
    LOG(info) << "No reading BadMap/Calibration from ccdb requested, set default";
    mHasCalib = true;
  }
  mCalibrator->setCuts(mPtMin, mEminHGTime, mEminLGTime, mEminHGTime, mEminLGTime);

  // Create geometry instance (inclusing reading mis-alignement)
  // instance will be pick up by Calibrator
  Geometry::GetInstance("Run3");
}

void PHOSEnergyCalibDevice::run(o2::framework::ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);

  // Do not use ccdb if already created
  if (!mHasCalib) { // Default map and calibration was not set, use CCDB
    LOG(info) << "Getting calib from CCDB";
    // update BadMap and calibration if necessary
    auto badMapPtr = pc.inputs().get<o2::phos::BadChannelsMap*>("bdmap");
    mCalibrator->setBadMap(badMapPtr.get());

    auto calibPtr = pc.inputs().get<o2::phos::CalibParams*>("clb");
    mCalibrator->setCalibration(calibPtr.get());
    mHasCalib = true;
  }
  mOutputDigits.clear();

  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("clusters").header)->startTime; // is this the timestamp of the current TF?
  const gsl::span<const Cluster>& clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");
  const gsl::span<const CluElement>& cluelements = pc.inputs().get<gsl::span<CluElement>>("cluelements");
  const gsl::span<const TriggerRecord>& cluTR = pc.inputs().get<gsl::span<TriggerRecord>>("clusterTriggerRecords");

  LOG(debug) << "[PHOSEnergyCalibDevice - run]  Received " << cluTR.size() << " TRs and " << clusters.size() << " clusters, running calibration";
  if (mRunStartTime == 0) {
    const auto ref = pc.inputs().getFirstValid(true);
    mRunStartTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->creation; // approximate time in ms
    mRunNumber = DataRefUtils::getHeader<o2::header::DataHeader*>(ref)->runNumber;

    auto LHCPeriodStr = pc.services().get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("LHCPeriod", "");
    if (!(LHCPeriodStr.empty())) {
      mLHCPeriod = LHCPeriodStr;
    } else {
      const char* months[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
      std::time_t tt = mRunStartTime / 1000; // ms->s
      std::tm* ltm = std::gmtime(&tt);
      mLHCPeriod = (std::string)months[ltm->tm_mon];
      LOG(warning) << "LHCPeriod is not available, using current month " << mLHCPeriod;
    }
  }
  mCalibrator->process(tfcounter, clusters, cluelements, cluTR, mOutputDigits);

  fillOutputTree();
}

void PHOSEnergyCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
  mCalibrator->endOfStream();
  if (mPostHistos) {
    postHistosCCDB(ec);
  }
  writeOutFile();
}

void PHOSEnergyCalibDevice::stop()
{
  // DDS stop method: simply write the latest tree
  LOG(info) << "STOP: writing output digits file";
  writeOutFile();
}
void PHOSEnergyCalibDevice::fillOutputTree()
{

  if (mOutputDigits.size() == 0) { // nothing to fill
    return;
  }

  LOG(info) << "Filling tree with " << mOutputDigits.size() << " digits";
  if (!mFileOut) { // create file and tree
    mFileName = mOutputDir + fmt::format("PHOS_CalibDigits_{}.root", mRunNumber);
    LOG(info) << "Creating new tree for PHOS calib digits, output file=" << mFileName;
    mFileOut = std::make_unique<TFile>(mFileName.c_str(), "recreate");
    mTreeOut = std::make_unique<TTree>("phosCalibDig", "O2 PHOS calib tree");
    mFileMetaData = std::make_unique<o2::dataformats::FileMetaData>();
  }
  auto* br = mTreeOut->GetBranch("PHOSCalib");
  auto* pptr = &mOutputDigits;
  if (br) {
    LOG(info) << " Using existing branch";
    br->SetAddress(&pptr);
  } else {
    LOG(info) << " Create new branch";
    br = mTreeOut->Branch("PHOSCalib", &pptr);
  }
  mTreeOut->Fill();
  br->ResetAddress();
}

void PHOSEnergyCalibDevice::writeOutFile()
{
  // write collected vector and metadata
  if (!mTreeOut) { // nothing to write,
    return;
  }
  LOG(info) << "Writing calibration digits";
  mFileOut->cd();
  int nbits = mTreeOut->Write();
  LOG(info) << "Wrote " << nbits << " bits";
  mTreeOut.reset();
  mFileOut->Close();
  mFileOut.reset();

  // write metaFile data
  mFileMetaData->fillFileData(mFileName);
  mFileMetaData->run = mRunNumber;
  mFileMetaData->LHCPeriod = mLHCPeriod;
  mFileMetaData->type = "calib";
  mFileMetaData->priority = "high";

  std::string metaFileNameTmp = mMetaFileDir + fmt::format("PHOS_CalibDigits_{}.tmp", mRunNumber);
  std::string metaFileName = mMetaFileDir + fmt::format("PHOS_CalibDigits_{}.done", mRunNumber);
  try {
    std::ofstream metaFileOut(metaFileNameTmp);
    metaFileOut << *mFileMetaData.get();
    metaFileOut.close();
    std::filesystem::rename(metaFileNameTmp, metaFileName);
  } catch (std::exception const& e) {
    LOG(error) << "Failed to store PHOS meta data file " << metaFileName << ", reason: " << e.what();
  }
  LOG(info) << "Stored metadate file " << mFileName << ".done";
  mFileMetaData.reset();
}
void PHOSEnergyCalibDevice::postHistosCCDB(o2::framework::EndOfStreamContext& ec)
{
  // prepare all info to be sent to CCDB
  auto flName = o2::ccdb::CcdbApi::generateFileName("TimeEnHistos");
  std::map<std::string, std::string> md;
  o2::ccdb::CcdbObjectInfo info("PHS/Calib/TimeEnHistos", "TimeEnHistos", flName, md, mRunStartTime, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
  info.setMetaData(md);
  auto image = o2::ccdb::CcdbApi::createObjectImage(mCalibrator->getCollectedHistos(), &info);

  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
  ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_TEHistos", subSpec}, *image.get());
  ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_TEHistos", subSpec}, info);
}
o2::framework::DataProcessorSpec o2::phos::getPHOSEnergyCalibDeviceSpec(bool useCCDB)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusters", o2::header::gDataOriginPHS, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cluelements", o2::header::gDataOriginPHS, "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusterTriggerRecords", o2::header::gDataOriginPHS, "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (useCCDB) {
    inputs.emplace_back("bdmap", o2::header::gDataOriginPHS, "PHS_BM", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/BadMap"));
    inputs.emplace_back("clb", o2::header::gDataOriginPHS, "PHS_Calibr", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/CalibParams"));
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHOS_TEHistos"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHOS_TEHistos"}, o2::framework::Lifetime::Sporadic);

  return o2::framework::DataProcessorSpec{"PHOSEnergyCalibDevice",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSEnergyCalibDevice>(useCCDB, ccdbRequest),
                                          o2::framework::Options{
                                            {"ptminmgg", o2::framework::VariantType::Float, 1.5f, {"minimal pt to fill mgg calib histos"}},
                                            {"eminhgtime", o2::framework::VariantType::Float, 1.5f, {"minimal E (GeV) to fill HG time calib histos"}},
                                            {"eminlgtime", o2::framework::VariantType::Float, 5.f, {"minimal E (GeV) to fill LG time calib histos"}},
                                            {"ecalibdigitmin", o2::framework::VariantType::Float, 0.05f, {"minimal digtit E (GeV) to keep digit for calibration"}},
                                            {"ecalibclumin", o2::framework::VariantType::Float, 0.4f, {"minimal cluster E (GeV) to keep digit for calibration"}},
                                            {"output-dir", VariantType::String, "./", {"ROOT trees output directory"}},
                                            {"meta-output-dir", VariantType::String, "./", {"metafile output directory"}}}};
}
