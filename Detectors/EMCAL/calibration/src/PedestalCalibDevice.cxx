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

#include "CommonUtils/VerbosityConfig.h"
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALBase/Geometry.h"
#include "Framework/TimingInfo.h"
#include "EMCALCalibration/PedestalCalibDevice.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/RawReaderMemory.h"
#include "DetectorsCalibration/Utils.h"
#include "EMCALCalib/CalibDB.h"
#include "Framework/ConcreteDataMatcher.h"
#include "CommonUtils/MemFileHelper.h"

#include <fairlogger/Logger.h>

using namespace o2::emcal;

void PedestalCalibDevice::init(o2::framework::InitContext& ctx)
{
  LOG(debug) << "[EMCALPedestalCalibDevice - init] Initialize converter ";
  if (!mGeometry) {
    mGeometry = Geometry::GetInstanceFromRunNumber(300000);
  }
  if (!mGeometry) {
    LOG(error) << "Failure accessing geometry";
  }

  resetStartTS();
}

void PedestalCalibDevice::run(o2::framework::ProcessingContext& ctx)
{
  if (!mRun) {
    const auto& tinfo = ctx.services().get<o2::framework::TimingInfo>();
    if (tinfo.runNumber != 0) {
      mRun = tinfo.runNumber;
    }
  }

  constexpr auto originEMC = o2::header::gDataOriginEMC;
  auto data = ctx.inputs().get<o2::emcal::PedestalProcessorData>(getPedDataBinding());
  LOG(debug) << "adding pedestal data";
  mPedestalData += data;
}

//________________________________________________________________
void PedestalCalibDevice::sendData(o2::framework::EndOfStreamContext& ec, const Pedestal& data) const
{
  LOG(info) << "sending pedestal data";
  constexpr auto originEMC = o2::header::gDataOriginEMC;

  std::map<std::string, std::string> md;
  auto clName = o2::utils::MemFileHelper::getClassName(data);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  o2::ccdb::CcdbObjectInfo objInfo(o2::emcal::CalibDB::getCDBPathChannelPedestals(), clName, flName, md, mStartTS, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP, true);

  auto image = o2::ccdb::CcdbApi::createObjectImage(&data, &objInfo);

  ec.outputs().snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_PEDCALIB", 0}, *image.get());
  ec.outputs().snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_PEDCALIB", 0}, objInfo);

  // the following goes to the DCS ccdb
  EMCALPedestalHelper helper;
  std::vector<char> vecPedData = helper.createPedestalInstruction(data, mAddRunNumber ? mRun : -1);
  if (mDumpToFile) {
    helper.dumpInstructions("EMCAL-Pedestals.txt", vecPedData);
  }

  auto clNameDCS = o2::utils::MemFileHelper::getClassName(vecPedData);
  auto flNameDCS = o2::ccdb::CcdbApi::generateFileName(clNameDCS);
  o2::ccdb::CcdbObjectInfo objInfoDCS("EMC/Calib/PDData", clNameDCS, flNameDCS, md, mStartTS, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP, true);
  auto imageDCS = o2::ccdb::CcdbApi::createObjectImage(&vecPedData, &objInfoDCS);

  ec.outputs().snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_PEDCALIBSTR", 1}, *imageDCS.get());
  ec.outputs().snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_PEDCALIBSTR", 1}, objInfoDCS);
}

//________________________________________________________________
void PedestalCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  Pedestal pedObj = mCalibExtractor.extractPedestals(mPedestalData);
  sendData(ec, pedObj);
  // reset Timestamp (probably not needed as program will probably be terminated)
  resetStartTS();
}

o2::framework::DataProcessorSpec o2::emcal::getPedestalCalibDevice(bool dumpToFile, bool addRunNum)
{

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back(o2::emcal::PedestalCalibDevice::getPedDataBinding(), o2::header::gDataOriginEMC, "PEDDATA", 0, o2::framework::Lifetime::Timeframe);
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_PEDCALIB"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_PEDCALIB"}, o2::framework::Lifetime::Sporadic);

  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_PEDCALIBSTR"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_PEDCALIBSTR"}, o2::framework::Lifetime::Sporadic);

  return o2::framework::DataProcessorSpec{
    "PedestalCalibrator",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::emcal::PedestalCalibDevice>(dumpToFile, addRunNum)},
    o2::framework::Options{}};
}