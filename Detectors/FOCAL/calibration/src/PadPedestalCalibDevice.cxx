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

#include <filesystem>
#include <string>

#include <TFile.h>
#include <TF1.h>
#include <TH1.h>

#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonConstants/Triggers.h"
#include "CommonUtils/MemFileHelper.h"
#include "DataFormatsFOCAL/Constants.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/WorkflowSpec.h"
#include "FOCALCalibration/PadPedestalCalibDevice.h"

using namespace o2::focal;

PadPedestalCalibDevice::PadPedestalCalibDevice(bool updateCCDB, const std::string& path, bool debugMode) : mUpdateCCDB(updateCCDB), mPath(path), mDebug(debugMode) {}

void PadPedestalCalibDevice::init(framework::InitContext& ctx)
{
  std::string histname, histtitle;
  for (auto layer = 0; layer < 18; layer++) {
    histname = "PADADC_Layer" + std::to_string(layer);
    histtitle = "Pad ADC distribution Layer " + std::to_string(layer);
    mADCDistLayer[layer] = std::make_unique<TH2D>(histname.data(), histtitle.data(), constants::PADLAYER_MODULE_NCHANNELS, -0.5, constants::PADLAYER_MODULE_NCHANNELS - 0.5, 2000, 0., 2000.);
    mADCDistLayer[layer]->SetDirectory(nullptr);
  }

  auto calibmethod = ctx.options().get<std::string>("calibmethod");
  if (calibmethod == "mean") {
    mExtractionMethod = Method_t::MEAN;
  } else if (calibmethod == "fit") {
    mExtractionMethod = Method_t::FIT;
  } else {
    LOG(error) << "Unknown calibration method - using default (mean) instead";
  }
  LOG(info) << "Writing calibration histograms in path: " << mPath;
}

void PadPedestalCalibDevice::run(framework::ProcessingContext& ctx)
{
  std::vector<char> rawbuffer;
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {
    if (rawData.header != nullptr && rawData.payload != nullptr) {
      const auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(rawData);

      gsl::span<const char> databuffer(rawData.payload, payloadSize);
      int currentpos = 0;
      int currentfee = 0;
      bool firstHBF = true;
      while (currentpos < databuffer.size()) {
        auto rdh = reinterpret_cast<const o2::header::RDHAny*>(databuffer.data() + currentpos);
        auto trigger = o2::raw::RDHUtils::getTriggerType(rdh);
        if (trigger & o2::trigger::HB) {
          // HB trigger received
          if (o2::raw::RDHUtils::getStop(rdh)) {
            // Data ready
            if (rawbuffer.size()) {
              // Only process if we actually have payload (skip empty HBF)
              if (currentfee == 0xcafe) { // Use FEE ID 0xcafe for PAD data
                processRawData(rawbuffer);
              }
            } else {
              LOG(debug) << "Payload size 0 - skip empty HBF";
            }
          } else {
            rawbuffer.clear();
            currentfee = o2::raw::RDHUtils::getFEEID(rdh);
          }
        }

        if (o2::raw::RDHUtils::getMemorySize(rdh) == o2::raw::RDHUtils::getHeaderSize(rdh)) {
          // Skip page if emtpy
          currentpos += o2::raw::RDHUtils::getOffsetToNext(rdh);
          continue;
        }

        // non-0 payload size:
        auto payloadsize = o2::raw::RDHUtils::getMemorySize(rdh) - o2::raw::RDHUtils::getHeaderSize(rdh);
        auto page_payload = databuffer.subspan(currentpos + o2::raw::RDHUtils::getHeaderSize(rdh), payloadsize);
        std::copy(page_payload.begin(), page_payload.end(), std::back_inserter(rawbuffer));
        currentpos += o2::raw::RDHUtils::getOffsetToNext(rdh);
      }
    }
  }
}

void PadPedestalCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  LOG(info) << "Data collected - calculating pedestals and sending them";
  calculatePedestals();
  sendData(ec.outputs());
}

void PadPedestalCalibDevice::sendData(o2::framework::DataAllocator& output)
{

  if (mUpdateCCDB) {
    // prepare all info to be sent to CCDB
    o2::ccdb::CcdbObjectInfo info;
    auto image = o2::ccdb::CcdbApi::createObjectImage(mPedestalContainer.get(), &info);

    auto flName = o2::ccdb::CcdbApi::generateFileName("Pedestals");
    info.setPath("FOC/Calib/PadPedestals");
    info.setObjectType("Pedestals");
    info.setFileName(flName);

    const auto now = std::chrono::system_clock::now();
    long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    info.setStartValidityTimestamp(timeStart);
    info.setEndValidityTimestamp(o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
    std::map<std::string, std::string> md;
    info.setMetaData(md);

    LOG(info) << "Sending object FOC/Calib/PadPedestals";

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(framework::Output{o2::calibration::Utils::gDataOriginCDBPayload, "FOC_PADPEDESTALS", subSpec}, *image.get());
    output.snapshot(framework::Output{o2::calibration::Utils::gDataOriginCDBWrapper, "FOC_PADPEDESTALS", subSpec}, info);
  }

  if (mDebug) {
    TFile pedestalWriter("FOCALPedestalCalib.root", "RECREATE");
    pedestalWriter.cd();
    pedestalWriter.WriteObjectAny(mPedestalContainer.get(), o2::focal::PadPedestal::Class(), "ccdb_object");
  }

  // store ADC distributions in local file
  std::filesystem::path filepath(mPath);
  filepath.append("FOCALPadPedestals.root");
  LOG(info) << "Writing ADC distribution to " << filepath.c_str();
  TFile distwriter(filepath.c_str(), "RECREATE");
  distwriter.cd();
  for (int layer = 0; layer < mADCDistLayer.size(); layer++) {
    mADCDistLayer[layer]->Write();
  }
}

void PadPedestalCalibDevice::processRawData(const gsl::span<const char> padWords)
{
  constexpr std::size_t EVENTSIZEPADGBT = 1180,
                        EVENTSIZECHAR = EVENTSIZEPADGBT * sizeof(PadGBTWord) / sizeof(char);
  auto nevents = padWords.size() / (EVENTSIZECHAR);
  for (int ievent = 0; ievent < nevents; ievent++) {
    gsl::span<const PadGBTWord> padWordsGBT(reinterpret_cast<const PadGBTWord*>(padWords.data()), padWords.size() / sizeof(PadGBTWord));
    mDecoder.reset();
    mDecoder.decodeEvent(padWordsGBT);
    // Fill histograms
    for (auto layer = 0; layer < 18; layer++) {
      auto layerdata = mDecoder.getData().getDataForASIC(layer).getASIC();
      for (std::size_t chan = 0; chan < constants::PADLAYER_MODULE_NCHANNELS; chan++) {
        mADCDistLayer[layer]->Fill(chan, layerdata.getChannel(chan).getADC());
      }
    }
  }
}

void PadPedestalCalibDevice::calculatePedestals()
{
  mPedestalContainer = std::make_unique<PadPedestal>();
  int nPedestals = 0;
  for (std::size_t layer = 0; layer < 18; layer++) {
    for (std::size_t channel = 0; channel < constants::PADLAYER_MODULE_NCHANNELS; channel++) {
      std::unique_ptr<TH1> channelADC(mADCDistLayer[layer]->ProjectionY("channelADC", channel + 1, channel + 1));
      try {
        mPedestalContainer->setPedestal(layer, channel, evaluatePedestal(channelADC.get()));
        nPedestals++;
      } catch (PadPedestal::InvalidChannelException& e) {
        LOG(error) << "Failure accessing setting pedestal: " << e.what();
      }
    }
  }
  LOG(info) << "Found pedestals for " << nPedestals << " channels";
}

double PadPedestalCalibDevice::evaluatePedestal(TH1* channel)
{
  double pedestal = 0;
  switch (mExtractionMethod) {
    case Method_t::MAX: {
      // Max model - get center of the maximum bin
      int maxbin = channel->GetMaximumBin();
      pedestal = channel->GetBinCenter(maxbin);
      break;
    }
    case Method_t::MEAN: {
      // Mean model - get the mean of the distribution
      pedestal = channel->GetMean();
      break;
    }
    case Method_t::FIT: {
      // Fit model - fit full distribution with Gauss and take mean
      TF1 pedmodal("pedmodel", "gaus", 0, 2000);
      channel->Fit(&pedmodal, "Q");
      pedestal = pedmodal.GetParameter(1);
      break;
    }
  }
  return pedestal;
}

o2::framework::DataProcessorSpec o2::focal::getPadPedestalCalibDevice(bool updateCCDB, const std::string& path, bool debug)
{
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "FOC_PADPEDESTALS"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "FOC_PADPEDESTALS"}, o2::framework::Lifetime::Sporadic);

  return o2::framework::DataProcessorSpec{"FOCALPadPedestalCalibDevice",
                                          o2::framework::select("A:FOC/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<o2::focal::PadPedestalCalibDevice>(updateCCDB, path, debug),
                                          o2::framework::Options{
                                            {"calibmethod", o2::framework::VariantType::String, "max", {"Method used for pedestal evaluation"}}}};
}