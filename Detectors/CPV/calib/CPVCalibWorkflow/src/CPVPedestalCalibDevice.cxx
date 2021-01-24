// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVCalibWorkflow/CPVPedestalCalibDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include <string>
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

using namespace o2::cpv;

void CPVPedestalCalibDevice::init(o2::framework::InitContext& ic)
{

  //Create histograms for mean and RMS
  short n = 3 * o2::cpv::Geometry::kNumberOfCPVPadsPhi * o2::cpv::Geometry::kNumberOfCPVPadsZ;
  mMean = std::unique_ptr<TH2F>(new TH2F("Mean", "Mean", n, 0.5, n + 0.5, 500, 0., 500.));
}

void CPVPedestalCalibDevice::run(o2::framework::ProcessingContext& ctx)
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
      // use the decoder to decode the raw data, and extract signals
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

void CPVPedestalCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(INFO) << "[CPVPedestalCalibDevice - endOfStream]";
  //calculate stuff here
  calculatePedestals();
  checkPedestals();
  sendOutput(ec.outputs());
}

void CPVPedestalCalibDevice::sendOutput(DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  // using clbUtils = o2::calibration::Utils;
  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    o2::ccdb::CcdbObjectInfo info;
    auto image = o2::ccdb::CcdbApi::createObjectImage(mPedestals.get(), &info);

    auto flName = o2::ccdb::CcdbApi::generateFileName("Pedestals");
    info.setPath("CPV/Calib/Pedestals");
    info.setObjectType("Pedestals");
    info.setFileName(flName);
    // TODO: should be changed to time of the run
    const auto now = std::chrono::system_clock::now();
    long timeStart = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    info.setStartValidityTimestamp(timeStart);
    info.setEndValidityTimestamp(99999999999999);
    std::map<std::string, std::string> md;
    info.setMetaData(md);

    LOG(INFO) << "Sending object CPV/Calib/Pedestals";

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBPayload, subSpec}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBInfo, subSpec}, info);
  }
  //Anyway send change to QC
  LOG(INFO) << "[CPVPedestalCalibDevice - run] Writing ";
  output.snapshot(o2::framework::Output{"CPV", "PEDDIFF", 0, o2::framework::Lifetime::Timeframe}, mPedDiff);

  //Write pedestal distributions to calculate bad map
  std::string filename = mPath + "CPVPedestals.root";
  TFile f(filename.data(), "RECREATE");
  mMean->Write();
  f.Close();
}

void CPVPedestalCalibDevice::calculatePedestals()
{

  mPedestals.reset(new Pedestals());

  //Calculate mean of pedestal distributions
  for (unsigned short i = mMean->GetNbinsX(); i > 0; i--) {
    TH1D* pr = mMean->ProjectionY(Form("proj%d", i), i, i);
    short pedMean = std::min(255, int(pr->GetMean()));
    pr->Delete();
    mPedestals->setPedestal(i - 1, pedMean);
  }
}

void CPVPedestalCalibDevice::checkPedestals()
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
  // for(short i=o2::cpv::Geometry::kNCHANNELS; --i;){
  //   short dp=mPedestals.getPedestal(i)-oldPed.getPedestal(i);
  //   mPedDiff[i]=dp ;
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

o2::framework::DataProcessorSpec o2::cpv::getPedestalCalibSpec(bool useCCDB, bool forceUpdate, std::string path)
{

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CPV", "PEDCALIBS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "PEDDIFF", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"PedestalCalibSpec",
                                          o2::framework::select("A:CPV/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<CPVPedestalCalibDevice>(useCCDB, forceUpdate, path),
                                          o2::framework::Options{}};
}
