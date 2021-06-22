// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSCalibWorkflow/PHOSHGLGRatioCalibDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include <string>
#include "FairLogger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/Cell.h"
#include "PHOSBase/Mapping.h"
#include <TFile.h>
#include <TF1.h>

using namespace o2::phos;

void PHOSHGLGRatioCalibDevice::init(o2::framework::InitContext& ic)
{

  //Create histograms for mean and RMS
  short n = o2::phos::Mapping::NCHANNELS - 1792;
  mhRatio.reset(new TH2F("HGLGRatio", "HGLGRatio", n, 1792.5, n + 1792.5, 100, 10., 20.));
  mCalibParams.reset(new CalibParams());
}

void PHOSHGLGRatioCalibDevice::run(o2::framework::ProcessingContext& ctx)
{

  // scan Cells stream, collect HG/LG pairs
  if (mRunStartTime == 0) {
    mRunStartTime = o2::header::get<o2::framework::DataProcessingHeader*>(ctx.inputs().get("cellTriggerRecords").header)->startTime;
  }

  auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
  LOG(DEBUG) << "[PHOSHGLGRatioCalibDevice - run]  Received " << cells.size() << " cells, running calibration ...";
  auto cellsTR = ctx.inputs().get<gsl::span<o2::phos::TriggerRecord>>("cellTriggerRecords");
  for (const auto& tr : cellsTR) {
    int firstCellInEvent = tr.getFirstEntry();
    int lastCellInEvent = firstCellInEvent + tr.getNumberOfObjects();
    mMapPairs.clear();

    for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
      const Cell c = cells[i];

      auto search = mMapPairs.find(c.getAbsId());
      if (search != mMapPairs.end()) { //exist
        if (c.getHighGain()) {
          search->second.mHGAmp = c.getEnergy();
        }
        if (c.getLowGain()) {
          search->second.mLGAmp = c.getEnergy();
        }
      } else {
        PairAmp pa = {0};
        if (c.getHighGain()) {
          pa.mHGAmp = c.getEnergy();
        }
        if (c.getLowGain()) {
          pa.mLGAmp = c.getEnergy();
        }
        mMapPairs.insert({c.getAbsId(), pa});
      }
    }
    fillRatios();
  }
}
void PHOSHGLGRatioCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(INFO) << "[PHOSHGLGRatioCalibDevice - endOfStream]";
  //calculate stuff here
  calculateRatios();
  checkRatios();
  sendOutput(ec.outputs());
}

void PHOSHGLGRatioCalibDevice::fillRatios()
{
  // scan collected map and fill ratio histogram if both channels are filled
  for (auto& it : mMapPairs) {
    if (it.second.mHGAmp > 0 && it.second.mLGAmp > mMinLG) {
      mhRatio->Fill(it.first, float(it.second.mHGAmp / it.second.mLGAmp));
    }
  }
  mMapPairs.clear(); //to avoid double counting
}

void PHOSHGLGRatioCalibDevice::calculateRatios()
{
  // Calculate mean of the ratio
  int n = o2::phos::Mapping::NCHANNELS;
  if (mhRatio->Integral() > 2 * minimalStatistics * n) { //average per channel

    TF1* fitFunc = new TF1("fitFunc", "gaus", 0., 4000.);
    fitFunc->SetParameters(1., 200., 60.);
    fitFunc->SetParLimits(1, 10., 2000.);
    for (int i = 1; i <= n; i++) {
      TH1D* tmp = mhRatio->ProjectionY(Form("channel%d", i), i, i);
      fitFunc->SetParameters(tmp->Integral(), tmp->GetMean(), tmp->GetRMS());
      if (tmp->Integral() < minimalStatistics) {
        tmp->Delete();
        continue;
      }
      tmp->Fit(fitFunc, "QL0", "", 0., 20.);
      float a = fitFunc->GetParameter(1);
      mCalibParams->setHGLGRatio(i, a); //absId starts from 0
      tmp->Delete();
    }
  }
}
void PHOSHGLGRatioCalibDevice::checkRatios()
{
  //Compare ratios to current ones stored in CCDB
  if (!mUseCCDB) {
    mUpdateCCDB = true;
    return;
  }
  LOG(INFO) << "Retrieving current HG/LG ratio from CCDB";
  //Read current padestals for comarison
  o2::ccdb::CcdbApi ccdb;
  ccdb.init(mCCDBPath); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> metadata;
  auto* currentCalibParams = ccdb.retrieveFromTFileAny<CalibParams>("PHS/Calib/CalibParams", metadata, mRunStartTime);

  if (!currentCalibParams) { //was not read from CCDB, but expected
    mUpdateCCDB = true;
    LOG(ERROR) << "Can not read current CalibParams from ccdb";
    return;
  }

  LOG(INFO) << "Got current calibration from CCDB";

  //Compare to current
  int nChanged = 0;
  for (short i = o2::phos::Mapping::NCHANNELS; i > 1792; i--) {
    short dp = 2;
    if (currentCalibParams->getHGLGRatio(i) > 0) {
      dp = mCalibParams->getHGLGRatio(i) / currentCalibParams->getHGLGRatio(i);
    }
    mRatioDiff[i] = dp;
    if (abs(dp - 1.) > 0.1) { //not a fluctuation
      nChanged++;
    }
    //Copy other stuff from the current CalibParams to new one
    mCalibParams->setGain(i, currentCalibParams->getGain(i));
    mCalibParams->setHGTimeCalib(i, currentCalibParams->getHGTimeCalib(i));
    mCalibParams->setLGTimeCalib(i, currentCalibParams->getLGTimeCalib(i));
  }
  LOG(INFO) << nChanged << "channels changed more that 1 ADC channel";
  if (nChanged > kMinorChange) { //serious change, do not update CCDB automatically, use "force" option to overwrite
    LOG(ERROR) << "too many channels changed: " << nChanged << " (threshold " << kMinorChange << ")";
    if (!mForceUpdate) {
      LOG(ERROR) << "you may use --forceupdate option to force updating ccdb";
    }
    mUpdateCCDB = false;
  } else {
    mUpdateCCDB = true;
  }
}

void PHOSHGLGRatioCalibDevice::sendOutput(DataAllocator& output)
{

  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    auto flName = o2::ccdb::CcdbApi::generateFileName("CalibParams");
    std::map<std::string, std::string> md;
    o2::ccdb::CcdbObjectInfo info("PHS/Calib/CalibParams", "CalibParams", flName, md, mRunStartTime, 99999999999999);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(mCalibParams.get(), &info);

    LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_HGLGratio", subSpec}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_HGLGratio", subSpec}, info);
  }
  //Anyway send change to QC
  LOG(INFO) << "[PHOSHGLGRatioCalibDevice - sendOutput] Sending QC ";
  output.snapshot(o2::framework::Output{"PHS", "CALIBDIFF", 0, o2::framework::Lifetime::Timeframe}, mRatioDiff);
}

DataProcessorSpec o2::phos::getHGLGRatioCalibSpec(bool useCCDB, bool forceUpdate, std::string path)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("cells", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLS"}, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLTRIGREC"}, o2::framework::Lifetime::Timeframe);

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginPHS, "CALIBDIFF", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHOS_HGLGratio"});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHOS_HGLGratio"});
  return o2::framework::DataProcessorSpec{"HGLGRatioCalibSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSHGLGRatioCalibDevice>(useCCDB, forceUpdate, path),
                                          o2::framework::Options{}};
}
