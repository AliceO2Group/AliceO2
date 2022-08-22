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

#ifndef O2_CALIBRATION_PHOSENERGY_CALIBDEV_H
#define O2_CALIBRATION_PHOSENERGY_CALIBDEV_H

/// @file   PHOSEnergyCalibDevice.h
/// @brief  Device to collect histos and digits for PHOS energy and time calibration.

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ProcessingContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataTakingContext.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "PHOSCalibWorkflow/PHOSEnergyCalibrator.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TFile.h"
#include "TTree.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSEnergyCalibDevice : public o2::framework::Task
{
 public:
  explicit PHOSEnergyCalibDevice(bool useCCDB, std::shared_ptr<o2::base::GRPGeomRequest> req, const std::string& outputDir, const std::string& metaFileDir, bool writeRootOutput) : mUseCCDB(useCCDB), mWriteRootOutput(writeRootOutput), mOutputDir(outputDir), mMetaFileDir(metaFileDir), mCCDBRequest(req) {}

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

  void stop() final;

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 protected:
  void updateTimeDependentParams(ProcessingContext& pc);
  void postHistosCCDB(o2::framework::EndOfStreamContext& ec);
  void fillOutputTree();
  void writeOutFile();

 private:
  static constexpr short kMaxCluInEvent = 64; /// maximal number of clusters per event to separate digits from them (6 bits in digit map)
  o2::framework::DataTakingContext mDataTakingContext{};
  bool mUseCCDB = false;
  bool mHasCalib = false;
  bool mPostHistos = true;      /// post colllected histos to ccdb
  bool mWriteRootOutput = true; /// Write local root files
  long mRunStartTime = 0;       /// start time of the run (sec)
  float mPtMin = 1.5;           /// minimal energy to fill inv. mass histo
  float mEminHGTime = 1.5;
  float mEminLGTime = 5.;
  float mEDigMin = 0.05;
  float mECluMin = 0.4;
  std::string mOutputDir;     /// where to write calibration digits
  std::string mFileName;      /// file name of output calib digits
  std::string mHistoFileName; /// file name of output calib digits
  std::string mMetaFileDir;   /// where to store meta files
  std::string mLHCPeriod;
  int mRunNumber = -1;
  std::unique_ptr<PHOSEnergyCalibrator> mCalibrator; /// Agregator of calibration TimeFrameSlots
  std::unique_ptr<const BadChannelsMap> mBadMap;     /// Latest bad channels map
  std::unique_ptr<const CalibParams> mCalibParams;   /// Latest bad channels map
  std::vector<uint32_t> mOutputDigits;               /// accumulated output digits
  std::unique_ptr<TFile> mFileOut;                   /// File to store output calib digits
  std::unique_ptr<TFile> mHistoFileOut;              /// File to store output histograms
  std::unique_ptr<TTree> mTreeOut;                   /// Tree to store output calib digits
  std::unique_ptr<o2::dataformats::FileMetaData> mFileMetaData;
  std::unique_ptr<o2::dataformats::FileMetaData> mHistoFileMetaData;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
};

o2::framework::DataProcessorSpec getPHOSEnergyCalibDeviceSpec(bool useCCDB, const std::string& outputDir, const std::string& metaFileDir, bool writeRootOutput);
} // namespace phos
} // namespace o2

#endif
