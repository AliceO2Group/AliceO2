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
  this->mCcdbUrlProd = ic.options().get<std::string>("ccdb-url-prod");
  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get DCSconfigObject_t from EPNs and aggregate them in 1 object
void ITSThresholdAggregator::run(ProcessingContext& pc)
{
  // Skip everything in case of garbage (potentially at EoS)
  if (pc.services().get<o2::framework::TimingInfo>().firstTForbit == -1U) {
    LOG(info) << "Skipping the processing of inputs for timeslice " << pc.services().get<o2::framework::TimingInfo>().timeslice << " (firstTForbit is " << pc.services().get<o2::framework::TimingInfo>().firstTForbit << ")";
    return;
  }
  // take run type, scan type, fit type, db version only at the beginning (important for EoS operations!)
  if (mRunType == -1) {
    for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "RUNT"}}})) {
      updateLHCPeriodAndRunNumber(pc);
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
    LOG(info) << "Aggregator received the following parameters:";
    LOG(info) << "Run type  : " << mRunType;
    LOG(info) << "Run number: " << mRunNumber;
    LOG(info) << "LHC period: " << mLHCPeriod;
    LOG(info) << "Scan type : " << mScanType;
    LOG(info) << "Fit type  : " << std::to_string(mFitType);
    LOG(info) << "DB version (no sense in pulse length 2D): " << mDBversion;
  }
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "TSTR"}}})) {
    // Read strings with tuning info
    const auto tunString = pc.inputs().get<gsl::span<char>>(inputRef);
    // Merge all strings coming from several sources (EPN)
    std::copy(tunString.begin(), tunString.end(), std::back_inserter(tuningMerge));
  }

  for (auto const& inputRef : InputRecordWalker(pc.inputs(), {{"check", ConcreteDataTypeMatcher{"ITS", "PIXTYP"}}})) {
    // Read strings with pixel type info
    const auto PixTypString = pc.inputs().get<gsl::span<char>>(inputRef);
    // Merge all strings coming from several sources (EPN)
    std::copy(PixTypString.begin(), PixTypString.end(), std::back_inserter(PIXTYPMerge));
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
                                                      : this->mFitType == 2 ? "hitcounting"
                                                                            : "null";
  if (mScanType == 'D' || mScanType == 'A' || mScanType == 'P' || mScanType == 'p') {
    ft = "null";
  }

  std::map<std::string, std::string> md = {
    {"fittype", ft}, {"runtype", std::to_string(this->mRunType)}, {"confDBversion", std::to_string(this->mDBversion)}};
  if (mScanType == 'p') {
    md["confDBversion"] = "null";
  }
  if (!(this->mLHCPeriod.empty())) {
    md.insert({"LHC_period", this->mLHCPeriod});
  }
  if (!mCcdbUrl.empty()) { // add only if we write here otherwise ccdb-populator-wf add it already
    md.insert({"runNumber", std::to_string(this->mRunNumber)});
  }

  if (!mCcdbUrlProd.empty()) { // add only if we write here otherwise ccdb-populator-wf add it already
    md.insert({"runNumber", std::to_string(this->mRunNumber)});
  }
  std::string path("ITS/Calib/");
  std::string name_str = mScanType == 'V' ? "VCASN" : mScanType == 'I' ? "ITHR"
                                                    : mScanType == 'D' ? "DIG"
                                                    : mScanType == 'A' ? "ANA"
                                                    : mScanType == 'T' ? "THR"
                                                    : mScanType == 'P' ? "PULSELENGTH"
                                                    : mScanType == 'p' ? "PULSELENGTH2D"
                                                                       : "NULL";
  o2::ccdb::CcdbObjectInfo info((path + name_str), "threshold_map", "calib_scan.root", md, tstart, tend);
  o2::ccdb::CcdbObjectInfo info_pixtyp((path + name_str), "threshold_map", "calib_scan.root", md, tstart, tend);

  auto image = o2::ccdb::CcdbApi::createObjectImage(&tuningMerge, &info);
  auto image_pixtyp = o2::ccdb::CcdbApi::createObjectImage(&PIXTYPMerge, &info_pixtyp);

  std::string file_name = "calib_scan_" + name_str + ".root";
  std::string file_name_pixtyp = "calib_scan_pixel_type_" + name_str + ".root";
  info.setFileName(file_name);
  info_pixtyp.setFileName(file_name_pixtyp);

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

      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "DIG", 1}, *image_pixtyp);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "DIG", 1}, info_pixtyp);
    } else if (this->mScanType == 'A') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ANA", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ANA", 0}, info);

      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ANA", 1}, *image_pixtyp);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ANA", 1}, info_pixtyp);
    } else if (this->mScanType == 'P') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PULSELENGTH", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PULSELENGTH", 0}, info);
    } else if (this->mScanType == 'p') {
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PULSELENGTH2D", 0}, *image);
      ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PULSELENGTH2D", 0}, info);
    } else {
      LOG(error) << "Nothing sent to ccdb-populator, mScanType" << mScanType << "does not match any known scan type";
    }
  }

  if (!mCcdbUrl.empty()) { // if url is specified, send object to ccdb from THIS wf

    LOG(info) << "Sending object " << info.getFileName() << " to " << mCcdbUrl << "/browse/" << info.getPath() << " from the ITS calib workflow";
    o2::ccdb::CcdbApi mApi;
    mApi.init(mCcdbUrl);
    mApi.storeAsBinaryFile(&image->at(0), image->size(), info.getFileName(), info.getObjectType(), info.getPath(),
                           info.getMetaData(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
  }

  if (!mCcdbUrlProd.empty()) { // if url is specified, send object to ccdb from THIS wf

    LOG(info) << "Sending Noisy, Dead and Inefficenct pixel object " << info_pixtyp.getFileName() << " (size:" << image_pixtyp->size() << ") to " << mCcdbUrlProd << "/browse/" << info_pixtyp.getPath() << " from the ITS calib workflow";
    o2::ccdb::CcdbApi mApiProd;
    mApiProd.init(mCcdbUrlProd);
    mApiProd.storeAsBinaryFile(&image_pixtyp->at(0), image_pixtyp->size(), info_pixtyp.getFileName(), info_pixtyp.getObjectType(), info_pixtyp.getPath(), info_pixtyp.getMetaData(), info_pixtyp.getStartValidityTimestamp(), info_pixtyp.getEndValidityTimestamp());
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
// Search current month  or LHCperiod, and read run number
void ITSThresholdAggregator::updateLHCPeriodAndRunNumber(ProcessingContext& pc)
{
  auto& dataTakingContext = pc.services().get<o2::framework::DataTakingContext>();
  const std::string LHCPeriodStr = dataTakingContext.lhcPeriod;
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

  // Run number
  auto& timingInfo = pc.services().get<o2::framework::TimingInfo>();
  this->mRunNumber = timingInfo.runNumber;

  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSThresholdAggregatorSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tunestring", ConcreteDataTypeMatcher{"ITS", "TSTR"});
  inputs.emplace_back("PixTypString", ConcreteDataTypeMatcher{"ITS", "PIXTYP"});
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

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "PULSELENGTH"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "PULSELENGTH"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "PULSELENGTH2D"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "PULSELENGTH2D"});

  return DataProcessorSpec{
    "its-aggregator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::its::ITSThresholdAggregator>()},
    Options{
      {"verbose", VariantType::Bool, false, {"Use verbose output mode"}},
      {"ccdb-url", VariantType::String, "", {"CCDB url, default is empty (i.e. no upload to CCDB)"}},
      {"ccdb-url-prod", VariantType::String, "", {"CCDB prod url, default is empty (i.e. no upload to CCDB)"}}}};
}
} // namespace its
} // namespace o2
