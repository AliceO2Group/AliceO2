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

#include "PHOSCalibWorkflow/PHOSHGLGRatioCalibDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include <string>
#include "FairLogger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/Cell.h"
#include "PHOSBase/Mapping.h"
#include <TFile.h>
#include <TF1.h>

using namespace o2::phos;

void PHOSHGLGRatioCalibDevice::init(o2::framework::InitContext& ic)
{

  mStatistics = ic.options().get<int>("statistics"); // desired number of events

  // Create histograms for mean and RMS
  short n = o2::phos::Mapping::NCHANNELS - 1792;
  mhRatio.reset(new TH2F("HGLGRatio", "HGLGRatio", n, 1792.5, n + 1792.5, 100, 10., 20.));
  mCalibParams.reset(new CalibParams());
}

void PHOSHGLGRatioCalibDevice::run(o2::framework::ProcessingContext& ctx)
{

  // scan Cells stream, collect HG/LG pairs
  if (mRunStartTime == 0) {
    const auto ref = ctx.inputs().getFirstValid(true);
    mRunStartTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->creation; // approximate time in ms
  }

  if (mStatistics <= 0) { // skip the rest of the run
    return;
  }
  if (!mOldCalibParams) { // Default map and calibration was not set, use CCDB
    LOG(info) << "Getting calib from CCDB";
    mOldCalibParams = ctx.inputs().get<o2::phos::CalibParams*>("oldclb");
  }

  auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
  LOG(debug) << "[PHOSHGLGRatioCalibDevice - run]  Received " << cells.size() << " cells, running calibration ...";
  auto cellsTR = ctx.inputs().get<gsl::span<o2::phos::TriggerRecord>>("cellTriggerRecords");
  for (const auto& tr : cellsTR) {
    int firstCellInEvent = tr.getFirstEntry();
    int lastCellInEvent = firstCellInEvent + tr.getNumberOfObjects();
    mMapPairs.clear();

    for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
      const Cell c = cells[i];

      auto search = mMapPairs.find(c.getAbsId());
      if (search != mMapPairs.end()) { // exist
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
    --mStatistics;
  }
  if (mStatistics <= 0) {
    LOG(info) << "[PHOSHGLGRatioCalibDevice - finalyzing calibration]";
    // calculate stuff here
    calculateRatios();
    checkRatios();
    sendOutput(ctx.outputs());
  }
}

void PHOSHGLGRatioCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  if (mStatistics > 0) { // not calculated yet
    LOG(info) << "[PHOSHGLGRatioCalibDevice - endOfStream]";
    // calculate stuff here
    calculateRatios();
    checkRatios();
    sendOutput(ec.outputs());
  }
}

void PHOSHGLGRatioCalibDevice::fillRatios()
{
  // scan collected map and fill ratio histogram if both channels are filled
  for (auto& it : mMapPairs) {
    if (it.second.mHGAmp > 0 && it.second.mLGAmp > mMinLG) {
      mhRatio->Fill(it.first, float(it.second.mHGAmp / it.second.mLGAmp));
    }
  }
  mMapPairs.clear(); // to avoid double counting
}

void PHOSHGLGRatioCalibDevice::calculateRatios()
{
  // Calculate mean of the ratio
  short n = o2::phos::Mapping::NCHANNELS - 1792;
  if (mhRatio->Integral() > 2 * minimalStatistics * n) { // average per channel

    TF1* fitFunc = new TF1("fitFunc", "gaus", 0., 4000.);
    fitFunc->SetParameters(1., 200., 60.);
    fitFunc->SetParLimits(1, 10., 2000.);
    for (short i = n; i > 0; i--) {
      TH1D* tmp = mhRatio->ProjectionY(Form("channel%d", i), i, i);
      fitFunc->SetParameters(tmp->Integral(), tmp->GetMean(), tmp->GetRMS());
      if (tmp->Integral() < minimalStatistics) {
        tmp->Delete();
        continue;
      }
      tmp->Fit(fitFunc, "QL0", "", 0., 20.);
      float a = fitFunc->GetParameter(1);
      mCalibParams->setHGLGRatio(i + 1792, a); // absId starts from 0
      tmp->Delete();
    }
  }
}

void PHOSHGLGRatioCalibDevice::checkRatios()
{
  // Compare ratios to current ones stored in CCDB
  if (!mUseCCDB) {
    mUpdateCCDB = true;
    // Set default values for gain and time
    for (short i = o2::phos::Mapping::NCHANNELS; i > 1792; i--) {
      mCalibParams->setGain(i, 0.005);
      mCalibParams->setHGTimeCalib(i, 0.);
      mCalibParams->setLGTimeCalib(i, 0.);
    }
    return;
  }
  LOG(info) << "Retrieving current HG/LG ratio from CCDB";

  if (!mOldCalibParams) { // was not read from CCDB, but expected
    mUpdateCCDB = true;
    LOG(error) << "Can not read current CalibParams from ccdb";
    return;
  }

  LOG(info) << "Got current calibration from CCDB";
  // Compare to current
  int nChanged = 0;
  for (short i = o2::phos::Mapping::NCHANNELS; i > 1792; i--) {
    short dp = 2;
    if (mOldCalibParams->getHGLGRatio(i) > 0) {
      dp = mCalibParams->getHGLGRatio(i) / mOldCalibParams->getHGLGRatio(i);
    } else {
      if (mCalibParams->getHGLGRatio(i) == 0 && mOldCalibParams->getHGLGRatio(i) == 0) {
        dp = 1.;
      }
    }
    mRatioDiff[i] = dp;
    if (abs(dp - 1.) > 0.1) { // not a fluctuation
      nChanged++;
    }
    // Copy other stuff from the current CalibParams to new one
    mCalibParams->setGain(i, mOldCalibParams->getGain(i));
    mCalibParams->setHGTimeCalib(i, mOldCalibParams->getHGTimeCalib(i));
    mCalibParams->setLGTimeCalib(i, mOldCalibParams->getLGTimeCalib(i));
  }
  LOG(info) << nChanged << "channels changed more than 10 %";
  if (nChanged > kMinorChange) { // serious change, do not update CCDB automatically, use "force" option to overwrite
    LOG(error) << "too many channels changed: " << nChanged << " (threshold " << kMinorChange << ")";
    if (!mForceUpdate) {
      LOG(error) << "you may use --forceupdate option to force updating ccdb";
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
    long validityTime = mRunStartTime + 31622400000; // one year validity range
    o2::ccdb::CcdbObjectInfo info("PHS/Calib/CalibParams", "CalibParams", flName, md, mRunStartTime, validityTime);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(mCalibParams.get(), &info);

    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_HGLGratio", subSpec}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_HGLGratio", subSpec}, info);
  }
  // Anyway send change to QC
  LOG(info) << "[PHOSHGLGRatioCalibDevice - sendOutput] Sending QC ";
  output.snapshot(o2::framework::Output{"PHS", "CALIBDIFF", 0, o2::framework::Lifetime::Timeframe}, mRatioDiff);
}

DataProcessorSpec o2::phos::getHGLGRatioCalibSpec(bool useCCDB, bool forceUpdate)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("cells", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLS"}, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLTRIGREC"}, o2::framework::Lifetime::Timeframe);
  if (useCCDB) {
    inputs.emplace_back("oldclb", o2::header::gDataOriginPHS, "PHS_Calibr", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/CalibParams"));
  }

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginPHS, "CALIBDIFF", 0, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHOS_HGLGratio"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHOS_HGLGratio"}, Lifetime::Sporadic);
  return o2::framework::DataProcessorSpec{"HGLGRatioCalibSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSHGLGRatioCalibDevice>(useCCDB, forceUpdate),
                                          o2::framework::Options{{"statistics", o2::framework::VariantType::Int, 100000, {"max. number of events to process"}}}};
}
