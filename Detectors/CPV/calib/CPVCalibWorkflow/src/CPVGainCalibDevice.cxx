// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVCalibWorkflow/CPVGainCalibDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include <string>
#include <ctime>
#include "FairLogger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/InputRecordWalker.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "CPVReconstruction/RawReaderMemory.h"
#include "CPVReconstruction/RawDecoder.h"
#include "CPVBase/Geometry.h"
#include "TF1.h"

using namespace o2::cpv;

void CPVGainCalibDevice::init(o2::framework::InitContext& ic)
{
  //Check if files from previous runs exist
  //if yes, read histogram
  mMean = std::unique_ptr<TH2F>(new TH2F("Gains", "Signals per channel", o2::cpv::Geometry::kNCHANNELS, 0.5, o2::cpv::Geometry::kNCHANNELS + 0.5, 1024, 0., 4096.));

  std::string filename = mPath + "CPVGains.root";
  TFile filein(filename.data(), "READ");
  if (filein.IsOpen()) {
    mMean->Add(static_cast<TH2F*>(filein.Get("Gains")));
    filein.Close();
  }
}

void CPVGainCalibDevice::run(o2::framework::ProcessingContext& ctx)
{

  //
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {

    o2::cpv::RawReaderMemory rawreader(o2::framework::DataRefUtils::as<const char>(rawData));
    // loop over all the DMA pages
    while (rawreader.hasNext()) {
      try {
        rawreader.next();
      } catch (RawErrorType_t e) {
        LOG(ERROR) << "Raw decoding error " << (int)e;
        //if problem in header, abandon this page
        if (e == RawErrorType_t::kPAGE_NOTFOUND ||
            e == RawErrorType_t::kHEADER_DECODING ||
            e == RawErrorType_t::kHEADER_INVALID) {
          break;
        }
        //if problem in payload, try to continue
        continue;
      }
      auto& header = rawreader.getRawHeader();
      auto triggerBC = o2::raw::RDHUtils::getTriggerBC(header);
      auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(header);
      // use the altro decoder to decode the raw data, and extract the RCU trailer
      o2::cpv::RawDecoder decoder(rawreader);
      RawErrorType_t err = decoder.decode();
      if (err != kOK) {
        //TODO handle severe errors
        continue;
      }
      // Loop over all the channels
      for (uint32_t adch : decoder.getDigits()) {
        AddressCharge ac = {adch};
        unsigned short absId = ac.Address;
        mMean->Fill(absId, ac.Charge);
      }
    } //RawReader::hasNext
  }
}

void CPVGainCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(INFO) << "[CPVGainCalibDevice - endOfStream]";
  //calculate stuff here
  calculateGains();
  checkGains();
  sendOutput(ec.outputs());
}

void CPVGainCalibDevice::sendOutput(DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  // using clbUtils = o2::calibration::Utils;
  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    o2::ccdb::CcdbObjectInfo info;
    auto image = o2::ccdb::CcdbApi::createObjectImage(mCalibParams.get(), &info);

    auto flName = o2::ccdb::CcdbApi::generateFileName("CalibParams");
    info.setPath("CPV/Calib/CalibParams");
    info.setObjectType("CalibParams");
    info.setFileName(flName);
    // TODO: should be changed to time of the run
    time_t now = time(nullptr);
    info.setStartValidityTimestamp(now);
    info.setEndValidityTimestamp(99999999999999);
    std::map<std::string, std::string> md;
    info.setMetaData(md);

    LOG(INFO) << "Sending object CPV/Calib/CalibParams";

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBPayload, subSpec}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBInfo, subSpec}, info);
  }

  //Write either final spectra (to calculate bad map) or temporary file
  if (mUpdateCCDB) { //good statistics, final spectra
    std::string filename = mPath + "CPVGains";
    time_t now = time(nullptr);
    tm* ltm = localtime(&now);
    filename += TString::Format("_%d%d%d%d%d.root", 1 + ltm->tm_min, 1 + ltm->tm_hour, ltm->tm_mday, 1 + ltm->tm_mon, 1970 + ltm->tm_year);
    LOG(DEBUG) << "opening file " << filename.data();
    TFile fout(filename.data(), "RECREATE");
    mMean->Write();
    fout.Close();
  } else {
    std::string filename = mPath + "CPVGains.root";
    LOG(INFO) << "statistics not sufficient yet: " << mMean->Integral() / mMean->GetNbinsX() << ", writing file " << filename;
    TFile fout(filename.data(), "RECREATE");
    mMean->Write();
    fout.Close();
  }
  //Anyway send change to QC
  output.snapshot(o2::framework::Output{"CPV", "GAINDIFF", 0, o2::framework::Lifetime::Timeframe}, mGainRatio);
}

void CPVGainCalibDevice::calculateGains()
{
  //Check if statistics is sufficient to fit distributions
  //Mean statistics should be ~2 times larger than minimal
  mUpdateCCDB = false;
  if (mMean->Integral() > 2 * kMinimalStatistics * (o2::cpv::Geometry::kNCHANNELS)) { //average per channel
    mCalibParams.reset(new CalibParams());

    TF1* fitFunc = new TF1("fitFunc", "landau", 0., 4000.);
    fitFunc->SetParameters(1., 200., 60.);
    fitFunc->SetParLimits(1, 10., 2000.);
    for (int i = 1; i <= mMean->GetNbinsX(); i++) {
      TH1D* tmp = mMean->ProjectionY(Form("channel%d", i), i, i);
      fitFunc->SetParameters(1., 200., 60.);
      if (tmp->Integral(20, 2000) < kMinimalStatistics) {
        tmp->Delete();
        continue;
      }
      tmp->Fit(fitFunc, "QL0", "", 20., 2000.);
      float a = fitFunc->GetParameter(1);
      if (a > 0) {
        a = 200. / a;
        mCalibParams->setGain(i - 1, a); //absId starts from 0
      }
      tmp->Delete();
    }
    mUpdateCCDB = true;
    //TODO: if file historam processed, remove temporary root file if it exists
  }
}

void CPVGainCalibDevice::checkGains()
{
  //Estimate if newly calculated gains are reasonable: close to reviously calculated
  // Do not update existing object automatically if difference is too strong
  // create object with validity range if far future (?) and send warning (e-mail?) to operator

  if (!mUpdateCCDB) { //gains were not calculated, do nothing
    return;
  }

  if (mUseCCDB) { // read calibration objects from ccdb
    // //TODO:
    // //Get current calibration
    // int nChanged=0;
    // for(short i=o2::cpv::Geometry::kNCHANNELS; --i;){
    //   short dp=2.
    // if(oldPed.getGain(i)>0) {
    //  dp = mCalibParams.getGain(i)/oldPed.getGain(i);
    // }
    //   mGainRatio[i]=dp ;
    //   if(abs(dp-1.)>0.1){ //not a fluctuation
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
}

o2::framework::DataProcessorSpec o2::cpv::getGainCalibSpec(bool useCCDB, bool forceUpdate, std::string path)
{

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CPV", "GAINCALIBS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "GAINDIFF", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"GainCalibSpec",
                                          o2::framework::select("A:CPV/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<CPVGainCalibDevice>(useCCDB, forceUpdate, path),
                                          o2::framework::Options{}};
}
