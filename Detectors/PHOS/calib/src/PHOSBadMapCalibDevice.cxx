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

#include "PHOSCalibWorkflow/PHOSBadMapCalibDevice.h"
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
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"

using namespace o2::phos;

void PHOSBadMapCalibDevice::init(o2::framework::InitContext& ic)
{

  mElowMin = ic.options().get<int>("ElowMin");
  mElowMax = ic.options().get<int>("ElowMax");
  mEhighMin = ic.options().get<int>("EhighMin");
  mEhighMax = ic.options().get<int>("EhighMax");

  short n = o2::phos::Mapping::NCHANNELS - 1792;
  if (mMode == 0) { //  Cell occupancy and time
    // Create histograms for lowE and high E occupancy and  Time
    mMeanLow.reset(new TH1F("MeanLowE", "Mean low E cut", n, 1792.5, n + 1792.5));
    mMeanHigh.reset(new TH1F("MeanHighE", "Mean high E cut", n, 1792.5, n + 1792.5));
    mMeanTime.reset(new TH1F("MeanTimeLow", "MeanTimeLow", n, 1792.5, n + 1792.5));
  }
  if (mMode == 1) { // chi2 distribution
    mChi2.reset(new TH1F("Chi2", "#chi^2", n, 1792.5, n + 1792.5));
    mChi2norm.reset(new TH1F("Chi2Norm", "#chi^2 normalization", n, 1792.5, n + 1792.5));
  }
  if (mMode == 2) { // Pedestals
    mHGMean.reset(new TH1F("HGMean", "High Gain mean", n, 1792.5, n + 1792.5));
    mHGRMS.reset(new TH1F("HGRMS", "High Gain RMS", n, 1792.5, n + 1792.5));
    mHGNorm.reset(new TH1F("HGNorm", "High Gain normalization", n, 1792.5, n + 1792.5));
    mLGMean.reset(new TH1F("LGMean", "Low Gain mean", n, 1792.5, n + 1792.5));
    mLGRMS.reset(new TH1F("LGRMS", "Low Gain RMS", n, 1792.5, n + 1792.5));
    mLGNorm.reset(new TH1F("LGNorm", "Low Gain normalization", n, 1792.5, n + 1792.5));
  }
}

void PHOSBadMapCalibDevice::run(o2::framework::ProcessingContext& ctx)
{
  if (mRunStartTime == 0) {
    if (mMode == 0 || mMode == 2) {
      mRunStartTime = o2::header::get<o2::framework::DataProcessingHeader*>(ctx.inputs().get("cells").header)->startTime;
    }
    if (mMode == 1) {
      mRunStartTime = o2::header::get<o2::framework::DataProcessingHeader*>(ctx.inputs().get("fitqa").header)->startTime;
    }
    mValidityTime = mRunStartTime + 31622400000; // one year validity range
  }

  // scan Cells stream, collect occupancy
  if (mMode == 0) { //  Cell occupancy and time
    auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
    LOG(debug) << "[PHOSBadMapCalibDevice - run]  Received " << cells.size() << " cells, running calibration ...";
    for (const auto& c : cells) {
      float e = c.getEnergy();
      if (e > mElowMin && e < mElowMax) {
        mMeanLow->Fill(c.getAbsId() - 1792);
      }
      if (e > mEhighMin && e < mEhighMax) {
        mMeanHigh->Fill(c.getAbsId() - 1792);
        mMeanTime->Fill(c.getAbsId() - 1792, c.getTime());
      }
    }
  } // mMode 0: cell occupancy

  if (mMode == 1) { // Raw data analysis, chi2
    auto chi2list = ctx.inputs().get<gsl::span<short>>("fitqa");
    LOG(info) << "[PHOSBadMapCalibDevice - run]  Received " << chi2list.size() << " Chi2, running calibration ...";
    auto b = chi2list.begin();
    while (b != chi2list.end()) {
      short absId = *b;
      ++b;
      float c = 0.2 * (*b);
      ++b;
      mChi2norm->Fill(absId - 1792);
      mChi2->Fill(absId - 1792, c);
    }
  }

  if (mMode == 2) { // Pedestals
    auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
    LOG(debug) << "[PHOSBadMapCalibDevice - run]  Received " << cells.size() << " cells, running calibration ...";
    for (const auto& c : cells) {
      float e = c.getEnergy();
      if (c.getHighGain()) {
        mHGMean->Fill(c.getAbsId() - 1792, c.getEnergy());
        mHGRMS->Fill(c.getAbsId() - 1792, 1.e+7 * c.getTime());
        mHGNorm->Fill(c.getAbsId() - 1792, 1.);
      } else {
        mLGMean->Fill(c.getAbsId() - 1792);
        mLGRMS->Fill(c.getAbsId() - 1792, 1.e+7 * c.getTime());
        mLGNorm->Fill(c.getAbsId() - 1792, 1.);
      }
    }
  }
}

void PHOSBadMapCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(info) << "[PHOSBadMapCalibDevice - endOfStream]";
  // calculate stuff here
  if (calculateBadMap()) {
    checkBadMap();
    sendOutput(ec.outputs());
  }
}

void PHOSBadMapCalibDevice::sendOutput(DataAllocator& output)
{

  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    std::string flName = o2::ccdb::CcdbApi::generateFileName("BadMap");
    std::string kind;
    if (mMode == 0) { // Occupancy
      kind = "PHS/Calib/BadMapOcc";
    }
    if (mMode == 1) { // Chi2
      kind = "PHS/Calib/BadMapChi";
    }
    if (mMode == 2) { // Pedestals
      kind = "PHS/Calib/BadMapPed";
    }
    std::map<std::string, std::string> md;
    o2::ccdb::CcdbObjectInfo info(kind, "BadMap", flName, md, mRunStartTime, mValidityTime);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(mBadMap.get(), &info);

    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_BadMap", subSpec}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_BadMap", subSpec}, info);
  }
  // Anyway send change to QC
  LOG(info) << "[PHOSBadMapCalibDevice - run] Sending QC ";
  output.snapshot(o2::framework::Output{"PHS", "CALIBDIFF", 0, o2::framework::Lifetime::Timeframe}, mBadMapDiff);
}

bool PHOSBadMapCalibDevice::calculateBadMap()
{
  mBadMap = std::make_unique<BadChannelsMap>();

  if (mMode == 0) { // occupancy
    // --check if statistics is sufficient
    // --prepare map of pairs (number of entries in cell, number of such cells)
    // --sort according number of entries
    // --remove pairs in the beginning and the end of the list till variation of mean becomes small
    if (mMeanLow->Integral() < 1.e+4) {
      LOG(error) << "Insufficient statistics: " << mMeanLow->Integral() << " entries in lowE histo, do nothing";
      return false;
    }

    float nMean, nRMS;
    calculateLimits(mMeanLow.get(), nMean, nRMS); // for low E occupamcy
    float nMinLow = std::max(float(1.), nMean - 6 * nRMS);
    float nMaxLow = nMean + 6 * nRMS;
    LOG(info) << "Limits for low E histo: " << nMinLow << "<n<" << nMaxLow;
    for (int i = 1; i <= mMeanLow->GetNbinsX(); i++) {
      int c = mMeanLow->GetBinContent(i);
      if (c < nMinLow || c > nMaxLow) {
        mBadMap->addBadChannel(i + 1792);
      }
    }
    calculateLimits(mMeanHigh.get(), nMean, nRMS); // for high
    float nMinHigh = std::max(float(1.), nMean - 6 * nRMS);
    float nMaxHigh = nMean + 6 * nRMS;
    LOG(info) << "Limits for high E histo: " << nMinHigh << "<n<" << nMaxHigh;
    for (int i = 1; i <= mMeanHigh->GetNbinsX(); i++) {
      int c = mMeanHigh->GetBinContent(i);
      if (c < nMinHigh || c > nMaxHigh) {
        mBadMap->addBadChannel(i + 1792);
      }
    }
  }

  if (mMode == 1) { // chi2 distribution
    if (mChi2norm->Integral() < 1.e+4) {
      LOG(error) << "Insufficient statistics: " << mChi2norm->Integral() << " entries in chi2Norm histo, do nothing";
      return false;
    }
    mChi2->Divide(mChi2norm.get());
    float nMean, nRMS;
    calculateLimits(mChi2.get(), nMean, nRMS); // for low E occupamcy
    float nMinHigh = std::max(float(0.), nMean - 6 * nRMS);
    float nMaxHigh = nMean + 6 * nRMS;
    for (int i = 1; i <= mChi2->GetNbinsX(); i++) {
      int c = mChi2->GetBinContent(i);
      if (c < nMinHigh || c > nMaxHigh) {
        mBadMap->addBadChannel(i + 1792);
      }
    }
  } // Chi2

  if (mMode == 2) { // Pedestals
    if (mHGNorm->Integral() < 2.e+4) {
      LOG(error) << "Insufficient statistics: " << mHGNorm->Integral() << " entries in mHGNorm histo, do nothing";
      return false;
    }
    float nMean, nRMS;
    mHGMean->Divide(mHGNorm.get());
    calculateLimits(mHGMean.get(), nMean, nRMS); // mean HG pedestals
    float nMinHigh = std::max(float(0.), nMean - 6 * nRMS);
    float nMaxHigh = nMean + 6 * nRMS;
    for (int i = 1; i <= mHGMean->GetNbinsX(); i++) {
      int c = mHGMean->GetBinContent(i);
      if (c < nMinHigh || c > nMaxHigh) {
        mBadMap->addBadChannel(i + 1792);
      }
    }
    mLGMean->Divide(mLGNorm.get());
    calculateLimits(mLGMean.get(), nMean, nRMS); // for low E occupamcy
    nMinHigh = std::max(float(0.), nMean - 6 * nRMS);
    nMaxHigh = nMean + 6 * nRMS;
    for (int i = 1; i <= mLGMean->GetNbinsX(); i++) {
      int c = mLGMean->GetBinContent(i);
      if (c < nMinHigh || c > nMaxHigh) {
        mBadMap->addBadChannel(i + 1792);
      }
    }
    mHGRMS->Divide(mHGNorm.get());
    calculateLimits(mHGRMS.get(), nMean, nRMS); // mean HG pedestals
    nMinHigh = std::max(float(0.), nMean - 6 * nRMS);
    nMaxHigh = nMean + 6 * nRMS;
    for (int i = 1; i <= mHGRMS->GetNbinsX(); i++) {
      int c = mHGRMS->GetBinContent(i);
      if (c < nMinHigh || c > nMaxHigh) {
        mBadMap->addBadChannel(i + 1792);
      }
    }
    mLGRMS->Divide(mHGNorm.get());
    calculateLimits(mLGRMS.get(), nMean, nRMS); // for low E occupamcy
    nMinHigh = std::max(float(0.), nMean - 6 * nRMS);
    nMaxHigh = nMean + 6 * nRMS;
    for (int i = 1; i <= mLGRMS->GetNbinsX(); i++) {
      int c = mLGRMS->GetBinContent(i);
      if (c < nMinHigh || c > nMaxHigh) {
        mBadMap->addBadChannel(i + 1792);
      }
    }

  } // Pedestals

  return true;
}
void PHOSBadMapCalibDevice::calculateLimits(TH1F* tmp, float& nMean, float& nRMS)
{

  // --prepare map of pairs (number of entries in cell, number of such cells)
  // --sort according number of entries
  // --remove pairs in the beginning and the end of the list till variation of mean becomes small

  std::map<short, int> histo{};
  for (int i = 1; i <= tmp->GetNbinsX(); i++) {
    int c = tmp->GetBinContent(i);
    if (c > 0) {
      histo[c] = histo[c] + 1;
    }
  }
  float mean = 0, rms = 0, nTot = 0, maxN = 0.;
  for (const auto& [n, nCells] : histo) {
    mean += n * nCells;
    rms += n * n * nCells;
    nTot += nCells;
    if (n > maxN) {
      maxN = n;
    }
  }
  const float eps = 0.05; // maximal variation of mean if remove outlyers
  // now remove largest entries till mean remains almost the same
  for (std::map<short, int>::reverse_iterator iter = histo.rbegin(); iter != histo.rend(); ++iter) {
    float nextMean = mean - iter->first * iter->second;
    float nextRMS = rms - iter->first * iter->first * iter->second;
    float nextTot = nTot - iter->second;
    if (nTot == 0. || nextTot == 0.) {
      break;
    }
    if (mean / nTot - nextMean / nextTot > eps * mean / nTot) {
      mean = nextMean;
      rms = nextRMS;
      nTot = nextTot;
    } else { // converged
      break;
    }
  }
  // now remove smallest entries till mean remains almost the same
  for (std::map<short, int>::iterator iter = histo.begin(); iter != histo.end(); ++iter) {
    float nextMean = mean - iter->first * iter->second;
    float nextRMS = rms - iter->first * iter->first * iter->second;
    float nextTot = nTot - iter->second;
    if (nTot == 0. || nextTot == 0.) {
      break;
    }
    if (mean / nTot - nextMean / nextTot > eps * mean / nTot) {
      mean = nextMean;
      rms = nextRMS;
      nTot = nextTot;
    } else { // converged
      break;
    }
  }
  // Now we have stable mean. calculate limits
  if (nTot > 0) {
    nMean = mean / nTot;
    rms /= nTot;
    nRMS = rms - nMean * nMean;
    if (rms > 0) {
      nRMS = sqrt(nRMS);
    } else {
      nRMS = 0.;
    }
  } else {
    nMean = 0.5 * maxN;
    nRMS = 0.5 * maxN;
  }
}

void PHOSBadMapCalibDevice::checkBadMap()
{
  if (!mUseCCDB) {
    mUpdateCCDB = true;
    return;
  }
  LOG(info) << "Retrieving current BadMap from CCDB";
  // Read current pedestals for comparison
  // Normally CCDB manager should get and own objects
  auto& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
  ccdbManager.setURL(o2::base::NameConf::getCCDBServer());
  LOG(info) << " set-up CCDB " << o2::base::NameConf::getCCDBServer();

  std::string kind;
  if (mMode == 0) { // Occupancy
    kind = "PHS/BadMapOcc";
  }
  if (mMode == 1) { // Chi2
    kind = "PHS/BadMapChi";
  }
  if (mMode == 2) { // Pedestals
    kind = "PHS/BadMapPed";
  }

  BadChannelsMap* badMap = ccdbManager.get<o2::phos::BadChannelsMap>(kind);

  if (!badMap) { // was not read from CCDB, but expected
    mUpdateCCDB = true;
    return;
  }

  LOG(info) << "Got current Pedestals from CCDB";

  // Compare to current
  int nChanged = 0;
  for (short i = o2::phos::Mapping::NCHANNELS; i > 1792; i--) {
    mBadMapDiff[i] = mBadMap->isChannelGood(i) - badMap->isChannelGood(i);
    if (mBadMapDiff[i] != 0) {
      nChanged++;
    }
  }
  LOG(info) << nChanged << " channels changed";
  if (nChanged > kMinorChange) { // serious change, do not update CCDB automatically, use "force" option to overwrite
    LOG(error) << "too many channels changed: " << nChanged;
    if (!mForceUpdate) {
      LOG(error) << "you may use --forceupdate option to force updating ccdb";
    }
    mUpdateCCDB = false;
  } else {
    mUpdateCCDB = true;
  }
}

o2::framework::DataProcessorSpec o2::phos::getBadMapCalibSpec(int mode)
{

  std::vector<InputSpec> inputs;
  if (mode == 0) {
    inputs.emplace_back("cells", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLS"}, o2::framework::Lifetime::Timeframe);
  }
  if (mode == 1) {
    inputs.emplace_back("fitqa", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLFITQA"}, o2::framework::Lifetime::Timeframe);
  }
  if (mode == 2) {
    inputs.emplace_back("cells", ConcreteDataTypeMatcher{o2::header::gDataOriginPHS, "CELLS"}, o2::framework::Lifetime::Timeframe);
  }

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginPHS, "CALIBDIFF", 0, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHS_BadMap"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHS_BadMap"}, o2::framework::Lifetime::Sporadic);
  return o2::framework::DataProcessorSpec{"BadMapCalibSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSBadMapCalibDevice>(mode),
                                          o2::framework::Options{
                                            {"ElowMin", o2::framework::VariantType::Int, 100, {"Low E minimum in ADC counts"}},
                                            {"ElowMax", o2::framework::VariantType::Int, 200, {"Low E maximum in ADC counts"}},
                                            {"EhighMin", o2::framework::VariantType::Int, 400, {"Low E minimum in ADC counts"}},
                                            {"EhighMax", o2::framework::VariantType::Int, 900, {"Low E maximum in ADC counts"}}}};
}
