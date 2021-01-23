// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVCalibWorkflow/CPVBadMapCalibDevice.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include <string>
#include "FairLogger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/InputRecordWalker.h"
#include "CPVReconstruction/RawReaderMemory.h"
#include "CPVReconstruction/RawDecoder.h"
#include <TFile.h>

using namespace o2::cpv;

void CPVBadMapCalibDevice::init(o2::framework::InitContext& ic)
{
}

void CPVBadMapCalibDevice::run(o2::framework::ProcessingContext& ctx)
{

  mBadMap.reset(new BadChannelMap());

  //Probably can be configured from configKeyValues?
  const float kMaxCut = 10.;
  const float kMinCut = 0.1;

  if (mMethod % 2 == 0) { //Gains: dead channels (medhod= 0,2)

    std::string filename = mPath + "CPVGains.root";
    TFile f(filename.data(), "READ");
    TH2F* spectra = nullptr;
    if (f.IsOpen()) {
      spectra = static_cast<TH2F*>(f.Get("Gains"));
    }
    if (!spectra) {
      LOG(ERROR) << "ERROR: can not read histo Gains from file " << filename.data();
      return;
    }
    float meanOccupancy = spectra->Integral() / spectra->GetNbinsX();
    short nBadChannels = 0;
    float improvedOccupancy = meanOccupancy;
    do {
      nBadChannels = 0;
      meanOccupancy = improvedOccupancy;
      improvedOccupancy = 0;
      short ngood = 0;
      for (unsigned short i = spectra->GetNbinsX(); --i;) {
        float a = spectra->Integral(i + 1, i + 1, 1, 1024);
        if (a > kMaxCut * meanOccupancy + 1 || a < kMinCut * meanOccupancy - 1 || a == 0) { //noisy or dead
          if (mBadMap->isChannelGood(i)) {
            mBadMap->addBadChannel(i);
            nBadChannels++;
          }
        } else {
          improvedOccupancy += a;
          ngood++;
        }
      }
      if (ngood > 0) {
        improvedOccupancy /= ngood;
      }
    } while (nBadChannels > 0);
    spectra->Delete();
    f.Close();
  }

  if (mMethod > 0) { //methods 1,2: use pedestals
    //Read latest pedestal file
    std::string filename = mPath + "CPVPedestals.root";
    TFile f(filename.data(), "READ");
    TH2F* pedestals = nullptr;
    if (f.IsOpen()) {
      pedestals = static_cast<TH2F*>(f.Get("Mean"));
    }
    if (!pedestals) {
      LOG(ERROR) << "ERROR: can not read histo Mean from file " << filename.data();
      return;
    }
    TH1D* proj = pedestals->ProjectionY("m");
    float meanPed = proj->GetMean();
    float rmsPed = proj->GetRMS();
    proj->Delete();
    short nBadChannels = 0;
    float improvedMean = meanPed, improvedRMS = rmsPed;
    do {
      nBadChannels = 0;
      meanPed = improvedMean;
      rmsPed = improvedRMS;
      improvedMean = 0.;
      improvedRMS = 0.;
      short ngood = 0;
      for (unsigned short i = pedestals->GetNbinsX(); --i;) {
        TH1D* pr = pedestals->ProjectionY(Form("proj%d", i), i + 1, i + 1);
        float prMean = pr->GetMean();
        float prRMS = pr->GetRMS();
        pr->Delete();
        if (prMean > kMaxCut * meanPed || prMean < kMinCut * meanPed || prMean == 0 ||
            prRMS > kMaxCut * rmsPed || prRMS < kMinCut * rmsPed) { //noisy or dead
          if (mBadMap->isChannelGood(i)) {
            mBadMap->addBadChannel(i);
            nBadChannels++;
          }
        } else {
          improvedMean += prMean;
          improvedRMS += prRMS;
          ngood++;
        }
      }
      if (ngood > 0) {
        improvedMean /= ngood;
        improvedRMS /= ngood;
      }
    } while (nBadChannels > 0);
    pedestals->Delete();
    f.Close();
  }

  if (!differFromCurrent() || mForceUpdate) {
    sendOutput(ctx.outputs());
  }
  ctx.services().get<ControlService>().endOfStream();
  ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

void CPVBadMapCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(INFO) << "[CPVBadMapCalibDevice - endOfStream]";
  //calculate stuff here
}

void CPVBadMapCalibDevice::sendOutput(DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  // using clbUtils = o2::calibration::Utils;

  if (mUpdateCCDB || mForceUpdate) {
    // prepare all info to be sent to CCDB
    o2::ccdb::CcdbObjectInfo info;
    auto image = o2::ccdb::CcdbApi::createObjectImage(mBadMap.get(), &info);

    auto flName = o2::ccdb::CcdbApi::generateFileName("BadChannelMap");
    info.setPath("CPV/Calib/BadChannelMap");
    info.setObjectType("BadChannelMap");
    info.setFileName(flName);
    // TODO: should be changed to time of the run
    time_t now = time(nullptr);
    info.setStartValidityTimestamp(now);
    info.setEndValidityTimestamp(99999999999999);
    std::map<std::string, std::string> md;
    info.setMetaData(md);

    LOG(INFO) << "Sending object CPV/Calib/BadChannelMap";

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBPayload, subSpec}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBInfo, subSpec}, info);
  }

  output.snapshot(o2::framework::Output{"CPV", "BADMAPCHANGE", 0, o2::framework::Lifetime::Timeframe}, mMapDiff);
}

bool CPVBadMapCalibDevice::differFromCurrent()
{
  // Method to compare new pedestals and latest in ccdb
  // Send difference to QC
  // Do not update existing object automatically if difference is too strong

  if (!mUseCCDB) { //can not compare, just update
    return false;
  }
  // read calibration objects from ccdb
  // int nSlots = pc.inputs().getNofParts(0);
  //       assert(pc.inputs().getNofParts(1) == nSlots);

  //       int lhcphaseIndex = -1;
  //       for (int isl = 0; isl < nSlots; isl++) {
  //         const auto wrp = pc.inputs().get<CcdbObjectInfo*>("clbInfo", isl);
  //         if (wrp->getStartValidityTimestamp() > tfcounter) { // replace tfcounter with the timestamp of the TF
  //           lhxphaseIndex = isl - 1;
  //           break;
  //         }
  //       }

  return false;
}

o2::framework::DataProcessorSpec o2::cpv::getBadMapCalibSpec(bool useCCDB, bool forceUpdate, std::string path, short method)
{

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CPV", "BADMAP", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "BADMAPCHANGE", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"BadMapCalibSpec",
                                          Inputs{},
                                          outputs,
                                          o2::framework::adaptFromTask<CPVBadMapCalibDevice>(useCCDB, forceUpdate, path, method),
                                          o2::framework::Options{}};
}
