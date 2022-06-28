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

/// @file   ThresholdAggregatorSpec.cxx

#include "ITSWorkflow/ThresholdAggregatorSpec.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace its
{

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSThresholdAggregator::ITSThresholdAggregator()
{
  mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSThresholdAggregator");
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdAggregator::init(InitContext& ic)
{
  LOGF(info, "ITSThresholdAggregator init...", mSelfName);

  this->mVerboseOutput = ic.options().get<bool>("verbose");
  this->mCcdbUrl = ic.options().get<std::string>("ccdb-url");

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get DCSconfigObject_t from EPNs and aggregate them in 1 object
void ITSThresholdAggregator::run(ProcessingContext& pc)
{
  // take run type and scan type at the beginning
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "RUNT"}}})) {
    if (mRunType == -1) {
      updateRunID(pc);
      updateLHCPeriod(pc);
    }
    mRunType = pc.inputs().get<short int>(inputRef);
    break;
  }
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "SCANT"}}})) {
    mScanType = pc.inputs().get<char>(inputRef);
    break;
  }
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "FITT"}}})) {
    mFitType = pc.inputs().get<char>(inputRef);
    break;
  }
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "CONFDBV"}}})) {
    mDBversion = pc.inputs().get<short int>(inputRef);
    break;
  }
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "TSTR"}}})) {
    // Read strings with tuning info
    const auto tunString = pc.inputs().get<gsl::span<char>>(inputRef);
    // Merge all strings coming from several sources (EPN)
    std::copy(tunString.begin(), tunString.end(), std::back_inserter(tuningMerge));
  }
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "QCSTR"}}})) {
    // Read strings with list of completed chips
    const auto chipDoneString = pc.inputs().get<gsl::span<char>>(inputRef);
    // Merge all strings coming from several sources (EPN)
    std::copy(chipDoneString.begin(), chipDoneString.end(), std::back_inserter(chipDoneMerge));
  }

  if (mVerboseOutput) {
    LOG(info) << "Chips completed: ";
    std::string tmpString(chipDoneMerge.begin(), chipDoneMerge.end());
    LOG(info) << tmpString;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdAggregator::finalize(EndOfStreamContext* ec)
{
  if (ec) {
    LOGF(info, "endOfStream report:", mSelfName);
  }

  // Below is CCDB stuff
  long tstart = o2::ccdb::getCurrentTimestamp();
  long tend = tstart + 365L * 24 * 3600 * 1000;

  auto class_name = o2::utils::MemFileHelper::getClassName(tuningMerge);

  // Create metadata for database object
  std::string ft = this->mFitType == 0 ? "derivative" : this->mFitType == 1 ? "fit"
                                                                            : "hitcounting";
  if (mScanType == 'D' || mScanType == 'A') {
    ft = "null";
  }

  std::map<std::string, std::string> md = {
    {"fittype", ft}, {"runtype", std::to_string(this->mRunType)}, {"confDBversion", std::to_string(this->mDBversion)}};
  if (!(this->mLHCPeriod.empty())) {
    md.insert({"LHC_period", this->mLHCPeriod});
  }
  if (!mCcdbUrl.empty()) { // add only if we write here otherwise ccdb-populator-wf add it already
    md.insert({"runNumber", std::to_string(this->mRunNumber)});
  }

  std::string path("ITS/Calib/");
  std::string name_str = mScanType == 'V' ? "VCASN" : mScanType == 'I' ? "ITHR"
                                                    : mScanType == 'D' ? "DIG"
                                                    : mScanType == 'A' ? "ANA"
                                                                       : "THR";
  o2::ccdb::CcdbObjectInfo info((path + name_str), "threshold_map", "calib_scan.root", md, tstart, tend);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&tuningMerge, &info);
  std::string file_name = "calib_scan_" + name_str + ".root";
  info.setFileName(file_name);

  if (ec) { // send to ccdb-populator wf only if there is an EndOfStreamContext
    LOG(info) << "Class Name: " << class_name << " | File Name: " << file_name
              << "\nSending to ccdb-populator the object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size() << " bytes, valid for "
              << info.getStartValidityTimestamp() << " : "
              << info.getEndValidityTimestamp();

    if (this->mScanType == 'V') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "VCASN", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "VCASN", 0}, info);
    } else if (this->mScanType == 'I') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITHR", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITHR", 0}, info);
    } else if (this->mScanType == 'T') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "THR", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "THR", 0}, info);
    } else if (this->mScanType == 'D') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "DIG", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "DIG", 0}, info);
    } else if (this->mScanType == 'A') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ANA", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ANA", 0}, info);
    } else {
      LOG(error) << "Nothing sent to ccdb-populator, mScanType does not match any known scan type";
    }
  }

  if (!mCcdbUrl.empty()) { // if url is specified, send object to ccdb from THIS wf

    LOG(info) << "Sending object " << info.getFileName() << " to " << mCcdbUrl << "/browse/" << info.getPath() << " from the ITS calib workflow";
    o2::ccdb::CcdbApi mApi;
    mApi.init(mCcdbUrl);
    mApi.storeAsBinaryFile(&image->at(0), image->size(), info.getFileName(), info.getObjectType(), info.getPath(),
                           info.getMetaData(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// fairMQ functionality; called automatically when the DDS stops processing
void ITSThresholdAggregator::stop()
{
  if (!mStopped) {
    this->finalize(nullptr);
    this->mStopped = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSThresholdAggregator::endOfStream(EndOfStreamContext& ec)
{
  if (!mStopped) {
    this->finalize(&ec);
    this->mStopped = true;
  }
  return;
}

/////////////////////////////////////////////////////////////////////////////
// Search current month  or LHCperiod
void ITSThresholdAggregator::updateLHCPeriod(ProcessingContext& pc)
{
  auto conf = pc.services().get<RawDeviceService>().device()->fConfig;
  const std::string LHCPeriodStr = conf->GetProperty<std::string>("LHCPeriod", "");
  if (!(LHCPeriodStr.empty())) {
    this->mLHCPeriod = LHCPeriodStr;
  } else {
    const char* months[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
    std::time_t now = std::time(nullptr);
    std::tm* ltm = std::gmtime(&now);
    this->mLHCPeriod = (std::string)months[ltm->tm_mon];
    LOG(warning) << "LHCPeriod is not available, using current month " << this->mLHCPeriod;
  }
  this->mLHCPeriod += "_ITS";

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Retrieve Run Number
void ITSThresholdAggregator::updateRunID(ProcessingContext& pc)
{
  const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(
    pc.inputs().getFirstValid(true));
  if (dh->runNumber != 0) {
    this->mRunNumber = dh->runNumber;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSThresholdAggregatorSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tunestring", ConcreteDataTypeMatcher{"ITS", "TSTR"});
  inputs.emplace_back("chipdonestring", ConcreteDataTypeMatcher{"ITS", "QCSTR"});
  inputs.emplace_back("runtype", ConcreteDataTypeMatcher{"ITS", "RUNT"});
  inputs.emplace_back("scantype", ConcreteDataTypeMatcher{"ITS", "SCANT"});
  inputs.emplace_back("fittype", ConcreteDataTypeMatcher{"ITS", "FITT"});
  inputs.emplace_back("confdbversion", ConcreteDataTypeMatcher{"ITS", "CONFDBV"});

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "VCASN"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "VCASN"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITHR"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITHR"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "THR"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "THR"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "DIG"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "DIG"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ANA"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ANA"});

  return DataProcessorSpec{
    "its-aggregator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::its::ITSThresholdAggregator>()},
    Options{
      {"verbose", VariantType::Bool, false, {"Use verbose output mode"}},
      {"ccdb-url", VariantType::String, "", {"CCDB url, default is empty (i.e. no upload to CCDB)"}}}};
}
} // namespace its
} // namespace o2
