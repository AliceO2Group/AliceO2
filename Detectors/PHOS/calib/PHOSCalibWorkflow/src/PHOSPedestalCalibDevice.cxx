// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSCalibWorkflow/PHOSPedestalCalibDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include <string>
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
#include <TFile.h>

using namespace o2::phos;

void PHOSPedestalCalibDevice::init(o2::framework::InitContext& ic)
{

  mMapping.reset(new o2::phos::Mapping(""));
  if (mMapping.get() == nullptr) {
    LOG(FATAL) << "Failed to initialize mapping";
  }
  if (mMapping->setMapping() != o2::phos::Mapping::kOK) {
    LOG(ERROR) << "Failed to construct mapping";
  }

  mRawFitter.reset(new o2::phos::CaloRawFitter());
  mRawFitter->setPedestal(); // work in pedestal evaluation mode

  //Create histograms for mean and RMS
  short n = o2::phos::Mapping::NCHANNELS;
  mMeanHG.reset(new TH2F("MeanHighGain", "MeanHighGain", n, 0.5, n + 0.5, 100, 0., 100.));
  mMeanLG.reset(new TH2F("MeanLowGain", "MeanLowGain", n, 0.5, n + 0.5, 100, 0., 100.));
  mRMSHG.reset(new TH2F("RMSHighGain", "RMSHighGain", n, 0.5, n + 0.5, 100, 0., 10.));
  mRMSLG.reset(new TH2F("RMSLowGain", "RMSLowGain", n, 0.5, n + 0.5, 100, 0., 10.));
}

void PHOSPedestalCalibDevice::run(o2::framework::ProcessingContext& ctx)
{

  //
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {

    o2::phos::RawReaderMemory rawreader(o2::framework::DataRefUtils::as<const char>(rawData));

    // loop over all the DMA pages
    while (rawreader.hasNext()) {
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
            //TODO: which results should be accepted? full configurable list
            for (int is = 0; is < mRawFitter->getNsamples(); is++) {
              if (caloFlag == Mapping::kHighGain) {
                mMeanHG->Fill(absId, mRawFitter->getAmp(is));
                mRMSHG->Fill(absId, mRawFitter->getTime(is));
              } else {
                mMeanLG->Fill(absId, mRawFitter->getAmp(is));
                mRMSLG->Fill(absId, mRawFitter->getTime(is));
              }
            }
          }
        }
      }
    } //RawReader::hasNext
  }
}

void PHOSPedestalCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(INFO) << "[PHOSPedestalCalibDevice - endOfStream]";
  //calculate stuff here
  //.........

  sendOutput(ec.outputs());
}

void PHOSPedestalCalibDevice::sendOutput(DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  // using clbUtils = o2::calibration::Utils;
  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    o2::ccdb::CcdbObjectInfo info;
    auto image = o2::ccdb::CcdbApi::createObjectImage(mPedestals.get(), &info);

    auto flName = o2::ccdb::CcdbApi::generateFileName("Pedestals");
    info.setPath("PHOS/Calib/Pedestals");
    info.setObjectType("Pedestals");
    info.setFileName(flName);
    // TODO: should be changed to time of the run
    const auto now = std::chrono::system_clock::now();
    long timeStart = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    info.setStartValidityTimestamp(timeStart);
    info.setEndValidityTimestamp(99999999999999);
    std::map<std::string, std::string> md;
    info.setMetaData(md);

    LOG(INFO) << "Sending object PHOS/Calib/Pedestals";

    // header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    // output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBPayload, subSpec}, *image.get());
    // output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBInfo, subSpec}, info);
  }
  //Anyway send change to QC
  LOG(INFO) << "[PHOSPedestalCalibDevice - run] Sending QC ";
  output.snapshot(o2::framework::Output{"PHS", "PEDDIFF", 0, o2::framework::Lifetime::Timeframe}, mPedHGDiff);
  output.snapshot(o2::framework::Output{"PHS", "PEDDIFF", 0, o2::framework::Lifetime::Timeframe}, mPedLGDiff);

  LOG(INFO) << "[PHOSPedestalCalibDevice - run] Writing ";
  //Write pedestal distributions to calculate bad map
  std::string filename = mPath + "PHOSPedestals.root";
  TFile f(filename.data(), "RECREATE");
  mMeanHG->Write();
  mMeanLG->Write();
  mRMSHG->Write();
  mRMSLG->Write();
  f.Close();
}

void PHOSPedestalCalibDevice::calculatePedestals()
{

  mPedestals.reset(new Pedestals());

  //Calculate mean of pedestal distributions
  for (unsigned short i = mMeanHG->GetNbinsX(); i > 0; i--) {
    TH1D* pr = mMeanHG->ProjectionY(Form("proj%d", i), i, i);
    float a = pr->GetMean();
    mPedestals->setHGPedestal(i - 1, std::min(255, int(a)));
    pr->Delete();
    pr = mMeanLG->ProjectionY(Form("projLG%d", i), i, i);
    a = pr->GetMean();
    mPedestals->setLGPedestal(i - 1, std::min(255, int(a)));
    pr->Delete();
  }
}

void PHOSPedestalCalibDevice::checkPedestals()
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
  //   short dp=mPedestals.getHGPedestal(i)-oldPed.getHGPedestal(i);
  //   mPedHGDiff[i]=dp ;
  //   if(abs(dp)>1){ //not a fluctuation
  //     nChanged++;
  //   }
  //     dp=mPedestals.getLGPedestal(i)-oldPed.getLGPedestal(i);
  //   mPedLGDiff[i]=dp ;
  //   if(abs(dp)>1){ //not a fluctuation
  //     nChanged++;
  //   }
  // }
  // if(nChanged>kMinorChange){ //serious change, do not update CCDB automatically, use "force" option to overwrite
  //   mUpdateCCDB=false;
  // }
  // else{
  //   mUpdateCCDB=true;
  // }
}

o2::framework::DataProcessorSpec o2::phos::getPedestalCalibSpec(bool useCCDB, bool forceUpdate, std::string path)
{

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("PHS", "PEDCALIBS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "PEDDIFF", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"PedestalCalibSpec",
                                          o2::framework::select("A:PHS/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSPedestalCalibDevice>(useCCDB, forceUpdate, path),
                                          o2::framework::Options{}};
}

// void list_files(const char *dirname="./", const char *pattern="collPHOS_*.root", std::vector<std::string> &vnames) {
//   TSystemDirectory dir(dirname, dirname);
//   TList *files = dir.GetListOfFiles();
//   if (files){
//     TSystemFile *file;
//     TString fname;
//     TIter next(files);
//     while ((file=(TSystemFile*)next())) {
//       fname = file->GetName();
//       if (!file->IsDirectory() && fname.Index(pattern) > -1) {
//         vnames.emplace_back(fname.Data()) ;
//       }
//     }
//   }
