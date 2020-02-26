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

#include "TOFWorkflow/CompressedInspectorTask.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/CompressedDataFormat.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

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
  mHistos1D["timebc"] = new TH1F("hTimeBC", ";time (24.4 ps)", 1024, 0., 1024.);
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
  for (auto& input : pc.inputs()) {

    /** input **/
    const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
    auto payloadIn = const_cast<char*>(input.payload);
    auto payloadInSize = headerIn->payloadSize;

    setDecoderBuffer(payloadIn);
    setDecoderBufferSize(payloadInSize);
    DecoderBase::run();
  }
}

void CompressedInspectorTask::headerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit)
{
  for (int ibit = 0; ibit < 11; ++ibit)
    if (crateHeader->slotEnableMask & (1 << ibit))
      mHistos2D["slotEnableMask"]->Fill(crateHeader->drmID, ibit + 2);
};

void CompressedInspectorTask::frameHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                                           const FrameHeader_t* frameHeader, const PackedHit_t* packedHits)
{
  mHistos1D["hHisto"]->Fill(frameHeader->numberOfHits);
  for (int i = 0; i < frameHeader->numberOfHits; ++i) {
    auto packedHit = packedHits + i;
    auto indexE = packedHit->channel +
                  8 * packedHit->tdcID +
                  120 * packedHit->chain +
                  240 * (frameHeader->trmID - 3) +
                  2400 * crateHeader->drmID;
    int time = packedHit->time;
    int timebc = time % 1024;
    time += (frameHeader->frameID << 13);

    mHistos1D["indexE"]->Fill(indexE);
    mHistos1D["time"]->Fill(time);
    mHistos1D["timebc"]->Fill(timebc);
    mHistos1D["tot"]->Fill(packedHit->tot);
  }
};

void CompressedInspectorTask::trailerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                                             const CrateTrailer_t* crateTrailer, const Diagnostic_t* diagnostics)
{
  for (int i = 0; i < crateTrailer->numberOfDiagnostics; ++i) {
    auto diagnostic = diagnostics + i;
    mHistos2D["diagnostic"]->Fill(crateHeader->drmID, diagnostic->slotID);
  }
};

} // namespace tof
} // namespace o2
