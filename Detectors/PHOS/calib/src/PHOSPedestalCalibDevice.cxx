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

#include "PHOSCalibWorkflow/PHOSPedestalCalibDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include <string>
#include <fairlogger/Logger.h>
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

using namespace o2::phos;

void PHOSPedestalCalibDevice::init(o2::framework::InitContext& ic)
{

  mStatistics = ic.options().get<int>("statistics"); // desired number of events
  LOG(info) << "PHOS pedestal init: will collect " << mStatistics << " events";
  LOG(info) << "mUseCCDB (try to get current object) = " << mUseCCDB;
  LOG(info) << "mForceUpdate (update CCDB anyway) =" << mForceUpdate;
  // Create histograms for mean and RMS
  short n = o2::phos::Mapping::NCHANNELS - 1792;
  mMeanHG.reset(new TH2F("MeanHighGain", "MeanHighGain", n, 1792.5, n + 1792.5, 100, 0., 100.));
  mMeanLG.reset(new TH2F("MeanLowGain", "MeanLowGain", n, 1792.5, n + 1792.5, 100, 0., 100.));
  mRMSHG.reset(new TH2F("RMSHighGain", "RMSHighGain", n, 1792.5, n + 1792.5, 100, 0., 10.));
  mRMSLG.reset(new TH2F("RMSLowGain", "RMSLowGain", n, 1792.5, n + 1792.5, 100, 0., 10.));
}

void PHOSPedestalCalibDevice::run(o2::framework::ProcessingContext& ctx)
{
  // scan Cells stream, collect mean and RMS then calculate average and post

  if (mRunStartTime == 0) {
    mRunStartTime = ctx.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
  }
  if (!mOldPed) { // Default map and calibration was not set, use CCDB
    LOG(info) << "Getting calib from CCDB";
    mOldPed = ctx.inputs().get<o2::phos::Pedestals*>("oldped");
  }

  if (mStatistics <= 0) { // skip the rest of the run
    return;
  }
  if (mStatistics % 100 == 0) {
    LOG(info) << mStatistics << " left to produce calibration";
  }
  auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
  LOG(debug) << "[PHOSPedestalCalibDevice - run]  Received " << cells.size() << " cells, running calibration ...";
  auto cellsTR = ctx.inputs().get<gsl::span<o2::phos::TriggerRecord>>("cellTriggerRecords");
  for (const auto& tr : cellsTR) {
    int firstCellInEvent = tr.getFirstEntry();
    int lastCellInEvent = firstCellInEvent + tr.getNumberOfObjects();
    for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
      const Cell c = cells[i];
      if (c.getHighGain()) {
        mMeanHG->Fill(c.getAbsId(), c.getEnergy());
        mRMSHG->Fill(c.getAbsId(), 1.e+7 * c.getTime());
      } else {
        mMeanLG->Fill(c.getAbsId(), c.getEnergy());
        mRMSLG->Fill(c.getAbsId(), 1.e+7 * c.getTime());
      }
    }
    --mStatistics;
  }
  if (mStatistics <= 0) {
    calculatePedestals();
    checkPedestals();
    sendOutput(ctx.outputs());
  }
}

void PHOSPedestalCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  if (mStatistics > 0) { // not calculated yet
    LOG(info) << "[PHOSPedestalCalibDevice - endOfStream]";
    // calculate stuff here
    calculatePedestals();
    checkPedestals();
    sendOutput(ec.outputs());
  }
}

void PHOSPedestalCalibDevice::sendOutput(DataAllocator& output)
{

  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    auto flName = o2::ccdb::CcdbApi::generateFileName("Pedestals");
    std::map<std::string, std::string> md;
    long validityTime = mRunStartTime + 7776000000; // 3 months validity time
    o2::ccdb::CcdbObjectInfo info("PHS/Calib/Pedestals", "Pedestals", flName, md, mRunStartTime, validityTime);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(mPedestals.get(), &info);

    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Pedestal", subSpec}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_Pedestal", subSpec}, info);
    // Now same for DCS as vector
    std::vector<short> dcsPedestals(2 * (o2::phos::Mapping::NCHANNELS - 1792));
    // copy HG then LG pedestals
    for (short absId = 1793; absId <= o2::phos::Mapping::NCHANNELS; absId++) {
      dcsPedestals.emplace_back(mPedestals->getHGPedestal(absId));
    }
    for (short absId = 1793; absId <= o2::phos::Mapping::NCHANNELS; absId++) {
      dcsPedestals.emplace_back(mPedestals->getLGPedestal(absId));
    }
    auto imageDCS = o2::ccdb::CcdbApi::createObjectImage(&dcsPedestals, &info);
    header::DataHeader::SubSpecificationType subSpec1{(header::DataHeader::SubSpecificationType)1};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Pedestal", subSpec1}, *imageDCS.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_Pedestal", subSpec1}, info);
  }
  // Anyway send change to QC
  LOG(info) << "[PHOSPedestalCalibDevice - run] Sending QC ";
  output.snapshot(o2::framework::Output{"PHS", "CALIBDIFF", 0, o2::framework::Lifetime::Timeframe}, mPedDiff);
}

void PHOSPedestalCalibDevice::calculatePedestals()
{
  LOG(info) << "Start pedestals calculation";

  mPedestals.reset(new Pedestals());

  // Calculate mean of pedestal distributions
  for (unsigned short i = mMeanHG->GetNbinsX(); i > 0; i--) {
    TH1D* pr = mMeanHG->ProjectionY(Form("proj%d", i), i, i);
    float a = pr->GetMean();
    short cellId = static_cast<short>(mMeanHG->GetXaxis()->GetBinCenter(i));
    mPedestals->setHGPedestal(cellId, std::min(255, int(a)));
    pr->Delete();
    pr = mMeanLG->ProjectionY(Form("projLG%d", i), i, i);
    a = pr->GetMean();
    mPedestals->setLGPedestal(cellId, std::min(255, int(a)));
    pr->Delete();
    pr = mRMSHG->ProjectionY(Form("projRMS%d", i), i, i);
    a = pr->GetMean();
    mPedestals->setHGRMS(cellId, a);
    pr->Delete();
    pr = mRMSLG->ProjectionY(Form("projRMSLG%d", i), i, i);
    a = pr->GetMean();
    mPedestals->setLGRMS(cellId, a);
    pr->Delete();
  }
  LOG(info) << "Pedestals calculated";
}

void PHOSPedestalCalibDevice::checkPedestals()
{
  if (!mUseCCDB) {
    mUpdateCCDB = true;
    return;
  }

  if (!mOldPed) { // was not read from CCDB, but expected
    mUpdateCCDB = true;
    return;
  }

  // Compare to current
  int nChanged = 0;
  for (short i = o2::phos::Mapping::NCHANNELS; i > 1792; i--) {
    short dp = mPedestals->getHGPedestal(i) - mOldPed->getHGPedestal(i);
    mPedDiff[i] = dp;
    if (abs(dp) > 1) { // not a fluctuation
      nChanged++;
    }
    dp = mPedestals->getLGPedestal(i) - mOldPed->getLGPedestal(i);
    mPedDiff[i + o2::phos::Mapping::NCHANNELS] = dp;
    if (abs(dp) > 1) { // not a fluctuation
      nChanged++;
    }
  }
  LOG(info) << nChanged << " channels changed more that 1 ADC channel";
  if (nChanged > kMinorChange) { // serious change, do not update CCDB automatically, use "force" option to overwrite
    LOG(error) << "too many channels changed: " << nChanged << " (threshold not more than " << kMinorChange << ")";
    if (!mForceUpdate) {
      LOG(error) << "you may use --forceupdate option to force updating ccdb";
    }
    mUpdateCCDB = false;
  } else {
    mUpdateCCDB = true;
  }
}

o2::framework::DataProcessorSpec o2::phos::getPedestalCalibSpec(bool useCCDB, bool forceUpdate)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("cells", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLS"}, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLTRIGREC"}, o2::framework::Lifetime::Timeframe);
  if (useCCDB) {
    inputs.emplace_back("oldped", o2::header::gDataOriginPHS, "PHS_Calibr", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/Pedestals"));
  }

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginPHS, "CALIBDIFF", 0, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHOS_Pedestal"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHOS_Pedestal"}, o2::framework::Lifetime::Sporadic);
  return o2::framework::DataProcessorSpec{"PedestalCalibSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSPedestalCalibDevice>(useCCDB, forceUpdate),
                                          o2::framework::Options{{"statistics", o2::framework::VariantType::Int, 1000, {"max. number of events to process"}}}};
}
