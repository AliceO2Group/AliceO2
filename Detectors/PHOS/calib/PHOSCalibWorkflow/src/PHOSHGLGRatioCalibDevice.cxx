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
#include "FairLogger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/InputRecordWalker.h"
#include "PHOSBase/Mapping.h"
#include "PHOSReconstruction/Bunch.h"
#include "PHOSReconstruction/AltroDecoder.h"
#include "PHOSReconstruction/RawDecodingError.h"
#include <string>
#include <TFile.h>
#include <TF1.h>

using namespace o2::phos;

void PHOSHGLGRatioCalibDevice::init(o2::framework::InitContext& ic)
{

  mMapping.reset(new o2::phos::Mapping(""));
  if (!mMapping) {
    LOG(FATAL) << "Failed to initialize mapping";
  }
  if (mMapping->setMapping() != o2::phos::Mapping::kOK) {
    LOG(ERROR) << "Failed to construct mapping";
  }
  mRawFitter.reset(new o2::phos::CaloRawFitter());

  //Create histograms for mean and RMS
  short n = o2::phos::Mapping::NCHANNELS;
  mhRatio.reset(new TH2F("HGLGRatio", "HGLGRatio", n, 0.5, n + 0.5, 100, 10., 20.));
}

void PHOSHGLGRatioCalibDevice::run(o2::framework::ProcessingContext& ctx)
{

  //
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {

    o2::phos::RawReaderMemory rawreader(o2::framework::DataRefUtils::as<const char>(rawData));

    // loop over all the DMA pages
    while (rawreader.hasNext()) {
      mMapPairs.clear();

      try {
        rawreader.next();
      } catch (RawDecodingError::ErrorType_t e) {
        LOG(ERROR) << "Raw decoding error " << (int)e;
        //if problem in header, abandon this page
        if (e == RawDecodingError::ErrorType_t::PAGE_NOTFOUND ||
            e == RawDecodingError::ErrorType_t::HEADER_DECODING ||
            e == RawDecodingError::ErrorType_t::HEADER_INVALID) {
          break;
        }
        //if problem in payload, try to continue
        continue;
      }
      auto& header = rawreader.getRawHeader();
      auto triggerBC = o2::raw::RDHUtils::getTriggerBC(header);
      auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(header);
      auto ddl = o2::raw::RDHUtils::getFEEID(header);

      o2::InteractionRecord currentIR(triggerBC, triggerOrbit);

      if (ddl > o2::phos::Mapping::NDDL) { //only 14 correct DDLs
        LOG(ERROR) << "DDL=" << ddl;
        continue; //skip STU ddl
      }
      // use the altro decoder to decode the raw data, and extract the RCU trailer
      o2::phos::AltroDecoder decoder(rawreader);
      AltroDecoderError::ErrorType_t err = decoder.decode();

      if (err != AltroDecoderError::kOK) {
        LOG(ERROR) << "Errror " << err << " in decoding DDL" << ddl;
      }
      auto& rcu = decoder.getRCUTrailer();
      auto& channellist = decoder.getChannels();
      // Loop over all the channels for this RCU
      for (auto& chan : channellist) {
        short absId;
        Mapping::CaloFlag caloFlag;
        short fee;
        Mapping::ErrorStatus s = mMapping->hwToAbsId(ddl, chan.getHardwareAddress(), absId, caloFlag);
        if (s != Mapping::ErrorStatus::kOK) {
          LOG(ERROR) << "Error in mapping"
                     << " ddl=" << ddl << " hwaddress " << chan.getHardwareAddress();
          continue;
        }
        if (caloFlag != Mapping::kTRU) { //HighGain or LowGain
          CaloRawFitter::FitStatus fitResults = mRawFitter->evaluate(chan.getBunches());
          if (fitResults == CaloRawFitter::FitStatus::kOK || fitResults == CaloRawFitter::FitStatus::kNoTime) {
            for (int is = 0; is < mRawFitter->getNsamples(); is++) {
              //if map entry already exist, fill second part, otherwise fill first map
              if (mRawFitter->isOverflow(is)) {
                continue;
              }
              auto search = mMapPairs.find(absId);
              if (search != mMapPairs.end()) { //exist
                if (caloFlag == Mapping::kHighGain) {
                  search->second.mHGAmp = mRawFitter->getAmp(is);
                }
                if (caloFlag == Mapping::kLowGain) {
                  search->second.mLGAmp = mRawFitter->getAmp(is);
                }

              } else {
                PairAmp pa = {0};
                if (caloFlag == Mapping::kHighGain) {
                  pa.mHGAmp = mRawFitter->getAmp(is);
                }
                if (caloFlag == Mapping::kLowGain) {
                  pa.mLGAmp = mRawFitter->getAmp(is);
                }
                mMapPairs.insert({absId, pa});
              }
            }
          }
        }
      }
      fillRatios();
    } //RawReader::hasNext
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
    mCalibParams.reset(new CalibParams());

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
      mCalibParams->setHGLGRatio(i - 1, a); //absId starts from 0
      tmp->Delete();
    }
  }
}
void PHOSHGLGRatioCalibDevice::checkRatios()
{
  //Compare pedestals to current ones stored in CCDB
  //and send difference to QC to check
  if (!mUseCCDB) {
    mUpdateCCDB = true;
    return;
  }
  // //TODO:
  // //Get current map
  // int nChanged=0;
  // for(short i=o2::phos::Mapping::NCHANNELS; --i;){
  //   short dp=2;
  //   if(oldPed.mCalibParams->setHGLGRatio(i)>0){
  //     dp = mCalibParams->setHGLGRatio(i)/oldPed.mCalibParams->setHGLGRatio(i);
  //   }
  //   mRatioDiff[i]=dp ;
  //   if(abs(dp-1.)>0.1){ //not a fluctuation
  //     nChanged++;
  //   }
  // if(nChanged>kMinorChange){ //serious change, do not update CCDB automatically, use "force" option to overwrite
  //   mUpdateCCDB=false;
  // }
  // else{
  //   mUpdateCCDB=true;
  // }
}

void PHOSHGLGRatioCalibDevice::sendOutput(DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  // using clbUtils = o2::calibration::Utils;
  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    o2::ccdb::CcdbObjectInfo info;
    auto image = o2::ccdb::CcdbApi::createObjectImage(mCalibParams.get(), &info);

    auto flName = o2::ccdb::CcdbApi::generateFileName("CalibParams");
    info.setPath("PHOS/Calib/CalibParams");
    info.setObjectType("CalibParams");
    info.setFileName(flName);
    // TODO: should be changed to time of the run
    const auto now = std::chrono::system_clock::now();
    long timeStart = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    info.setStartValidityTimestamp(timeStart);
    info.setEndValidityTimestamp(99999999999999);
    std::map<std::string, std::string> md;
    info.setMetaData(md);

    LOG(INFO) << "Sending object PHOS/Calib/CalibParams";

    // header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    // output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBPayload, subSpec}, *image.get());
    // output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBInfo, subSpec}, info);
  }
  //Anyway send change to QC
  LOG(INFO) << "[PHOSPedestalCalibDevice - run] Writing ";
  output.snapshot(o2::framework::Output{"PHS", "HGLGRATIODIFF", 0, o2::framework::Lifetime::Timeframe}, mRatioDiff);

  //Write pedestal distributions to calculate bad map
  std::string filename = mPath + "PHOSHGLHRatio.root";
  TFile f(filename.data(), "RECREATE");
  mhRatio->Write();
  f.Close();
}

DataProcessorSpec o2::phos::getHGLGRatioCalibSpec(bool useCCDB, bool forceUpdate, std::string path)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("RAWDATA", o2::framework::ConcreteDataTypeMatcher{"PHS", "RAWDATA"}, o2::framework::Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("PHS", "HGLGRATIO", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "HGLGRATIODIFF", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"HGLGRatioCalibSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::PHOSHGLGRatioCalibDevice>(useCCDB, forceUpdate, path),
                                          o2::framework::Options{}};
}
