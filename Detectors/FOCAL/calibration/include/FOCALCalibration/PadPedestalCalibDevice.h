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
#ifndef ALICEO2_FOCAL_PADPEDESTALCALIBDEVICE_H
#define ALICEO2_FOCAL_PADPEDESTALCALIBDEVICE_H

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <gsl/span>
#include <TH2.h>
#include "Framework/DataAllocator.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "FOCALCalib/PadPedestal.h"
#include "FOCALReconstruction/PadDecoder.h"

namespace o2::focal
{

class PadPedestalCalibDevice : public framework::Task
{
 public:
  enum Method_t {
    MAX,
    MEAN,
    FIT
  };
  PadPedestalCalibDevice(bool updateCCDB, const std::string& path, bool debugMode);
  ~PadPedestalCalibDevice() final = default;

  void init(framework::InitContext& ctx) final;
  void run(framework::ProcessingContext& ctx) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void sendData(o2::framework::DataAllocator& output);
  void calculatePedestals();
  double evaluatePedestal(TH1* channel);
  void processRawData(const gsl::span<const char> padWords);

  PadDecoder mDecoder;
  std::unique_ptr<PadPedestal> mPedestalContainer;
  bool mUpdateCCDB = true;
  bool mDebug = false;
  std::string mPath = "./";
  Method_t mExtractionMethod = Method_t::MAX;
  std::array<std::unique_ptr<TH2>, 18> mADCDistLayer;
};

o2::framework::DataProcessorSpec getPadPedestalCalibDevice(bool updateCCDB, const std::string& path, bool debug);

} // namespace o2::focal

#endif