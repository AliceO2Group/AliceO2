// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedInspectorTask.cxx
/// @author Roberto Preghenella
/// @since  2020-01-25
/// @brief  TOF compressed data inspector task

#include "TOFCompression/CompressedInspectorTask.h"
#include "TOFCompression/RawDataFrame.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/CompressedDataFormat.h"
#include "DetectorsRaw/HBFUtils.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

//#define VERBOSE

using namespace o2::framework;

namespace o2
{
namespace tof
{

void CompressedInspectorTask::init(InitContext& ic)
{
  LOG(INFO) << "CompressedInspector init";
  auto filename = ic.options().get<std::string>("tof-inspector-filename");

  /** open file **/
  if (mFile && mFile->IsOpen()) {
    LOG(WARNING) << "a file was already open, closing";
    mFile->Close();
    delete mFile;
  }
  mFile = TFile::Open(filename.c_str(), "RECREATE");
  if (!mFile || !mFile->IsOpen()) {
    LOG(ERROR) << "cannot open output file: " << filename;
    mStatus = true;
    return;
  }

  mHistos1D["hHisto"] = new TH1F("hHisto", "", 1000, 0., 1000.);
  mHistos1D["time"] = new TH1F("hTime", ";time (24.4 ps)", 2097152, 0., 2097152.);
  mHistos1D["tot"] = new TH1F("hTOT", ";ToT (48.8 ps)", 2048, 0., 2048.);
  mHistos1D["indexE"] = new TH1F("hIndexE", ";index EO", 172800, 0., 172800.);
  mHistos2D["slotEnableMask"] = new TH2F("hSlotEnableMask", ";crate;slot", 72, 0., 72., 12, 1., 13.);
  mHistos2D["diagnostic"] = new TH2F("hDiagnostic", ";crate;slot", 72, 0., 72., 12, 1., 13.);

  auto finishFunction = [this]() {
    LOG(INFO) << "CompressedInspector finish";
    for (auto& histo : mHistos1D)
      histo.second->Write();
    for (auto& histo : mHistos2D)
      histo.second->Write();
    mFile->Close();
  };
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
  //  ic.services().get<CallbackService>().set(CallbackService::Id::EndOfStream, finishFunction);
}

void CompressedInspectorTask::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "CompressedInspector run";

  /** check status **/
  if (mStatus) {
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  /** receive input **/
  auto dataFrame = pc.inputs().get<RawDataFrame*>("dataframe");
  auto pointer = dataFrame->mBuffer;

  /** process input **/
  while (pointer < (dataFrame->mBuffer + dataFrame->mSize)) {
    auto rdh = reinterpret_cast<o2::header::RAWDataHeader*>(pointer);

    /** RDH close detected **/
    if (rdh->stop) {
#ifdef VERBOSE
      std::cout << "--- RDH close detected" << std::endl;
      o2::raw::HBFUtils::printRDH(*rdh);
#endif
      pointer += rdh->offsetToNext;
      continue;
    }

#ifdef VERBOSE
    std::cout << "--- RDH open detected" << std::endl;
    o2::raw::HBFUtils::printRDH(*rdh);
#endif

    pointer += rdh->headerSize;

    while (pointer < (reinterpret_cast<char*>(rdh) + rdh->memorySize)) {

      auto word = reinterpret_cast<uint32_t*>(pointer);
      if ((*word & 0x80000000) != 0x80000000) {
        printf(" %08x [ERROR] \n ", *(uint32_t*)pointer);
        return;
      }

      /** crate header detected **/
      auto crateHeader = reinterpret_cast<compressed::CrateHeader_t*>(pointer);
#ifdef VERBOSE
      printf(" %08x CrateHeader          (drmID=%d) \n ", *(uint32_t*)pointer, crateHeader->drmID);
#endif
      for (int ibit = 0; ibit < 11; ++ibit)
        if (crateHeader->slotEnableMask & (1 << ibit))
          mHistos2D["slotEnableMask"]->Fill(crateHeader->drmID, ibit + 2);
      pointer += 4;

      /** crate orbit expected **/
      auto crateOrbit = reinterpret_cast<compressed::CrateOrbit_t*>(pointer);
#ifdef VERBOSE
      printf(" %08x CrateOrbit           (orbit=0x%08x) \n ", *(uint32_t*)pointer, crateOrbit->orbitID);
#endif
      pointer += 4;

      while (true) {
        word = reinterpret_cast<uint32_t*>(pointer);

        /** crate trailer detected **/
        if (*word & 0x80000000) {
          auto crateTrailer = reinterpret_cast<compressed::CrateTrailer_t*>(pointer);
#ifdef VERBOSE
          printf(" %08x CrateTrailer         (numberOfDiagnostics=%d) \n ", *(uint32_t*)pointer, crateTrailer->numberOfDiagnostics);
#endif
          pointer += 4;

          /** loop over diagnostics **/
          for (int i = 0; i < crateTrailer->numberOfDiagnostics; ++i) {
            auto diagnostic = reinterpret_cast<compressed::Diagnostic_t*>(pointer);
#ifdef VERBOSE
            printf(" %08x Diagnostic           (slotId=%d) \n ", *(uint32_t*)pointer, diagnostic->slotID);
#endif
            mHistos2D["diagnostic"]->Fill(crateHeader->drmID, diagnostic->slotID);
            pointer += 4;
          }

          break;
        }

        /** frame header detected **/
        auto frameHeader = reinterpret_cast<compressed::FrameHeader_t*>(pointer);
#ifdef VERBOSE
        printf(" %08x FrameHeader          (numberOfHits=%d) \n ", *(uint32_t*)pointer, frameHeader->numberOfHits);
#endif
        mHistos1D["hHisto"]->Fill(frameHeader->numberOfHits);
        pointer += 4;

        /** loop over hits **/
        for (int i = 0; i < frameHeader->numberOfHits; ++i) {
          auto packedHit = reinterpret_cast<compressed::PackedHit_t*>(pointer);
#ifdef VERBOSE
          printf(" %08x PackedHit            (tdcID=%d) \n ", *(uint32_t*)pointer, packedHit->tdcID);
#endif
          auto indexE = packedHit->channel +
                        8 * packedHit->tdcID +
                        120 * packedHit->chain +
                        240 * (frameHeader->trmID - 3) +
                        2400 * crateHeader->drmID;
          int time = packedHit->time;
          time += (frameHeader->frameID << 13);

          mHistos1D["indexE"]->Fill(indexE);
          mHistos1D["time"]->Fill(time);
          mHistos1D["tot"]->Fill(packedHit->tot);
          pointer += 4;
        }
      }
    }

    pointer = reinterpret_cast<char*>(rdh) + rdh->offsetToNext;
  }

  /** write to file **/
  //  mFile.write(dataFrame->mBuffer, dataFrame->mSize);
}

DataProcessorSpec CompressedInspectorTask::getSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tof-compressed-inspector",
    Inputs{InputSpec("dataframe", o2::header::gDataOriginTOF, "CMPDATAFRAME", 0, Lifetime::Timeframe)},
    Outputs{},
    AlgorithmSpec{adaptFromTask<CompressedInspectorTask>()},
    Options{
      {"tof-inspector-filename", VariantType::String, "inspector.root", {"Name of the inspector output file"}}}};
}

} // namespace tof
} // namespace o2
