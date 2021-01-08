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

#include "TOFWorkflowUtils/CompressedInspectorTask.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
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

template <typename RDH>
void CompressedInspectorTask<RDH>::init(InitContext& ic)
{
  LOG(INFO) << "CompressedInspector init";
  auto filename = ic.options().get<std::string>("tof-compressed-inspector-filename");
  auto verbose = ic.options().get<bool>("tof-compressed-inspector-decoder-verbose");

  DecoderBaseT<RDH>::setDecoderVerbose(verbose);

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
  mHistos2D["slotPartMask"] = new TH2F("hSlotPartMask", ";crate;slot", 72, 0., 72., 12, 1., 13.);
  mHistos2D["diagnostic"] = new TH2F("hDiagnostic", ";crate;slot", 72, 0., 72., 13, 0., 13.);
  mHistos1D["Nerror"] = new TH1F("hNError", ";number of error", 1000, 0., 1000.);
  mHistos1D["Ntest"] = new TH1F("hNTest", ";number of test", 1000, 0., 1000.);
  mHistos1D["errorBit"] = new TH1F("hErrorBit", ";TDC error bit", 15, 0., 15.);
  mHistos2D["error"] = new TH2F("hError", ";slot;TDC", 24, 1., 13., 15, 0., 15.);
  mHistos2D["test"] = new TH2F("hTest", ";slot;TDC", 24, 1., 13., 15, 0., 15.);
  mHistos2D["crateBC"] = new TH2F("hCrateBC", ";crate;BC", 72, 0., 72., 4096, 0., 4096.);
  mHistos2D["crateOrbit"] = new TH2F("hCrateOrbit", ";crate;orbit", 72, 0., 72., 4096, 0., 4096.);

  auto finishFunction = [this]() {
    LOG(INFO) << "CompressedInspector finish";
    for (auto& histo : mHistos1D) {
      histo.second->Write();
    }
    for (auto& histo : mHistos2D) {
      histo.second->Write();
    }
    mFile->Close();
  };
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

template <typename RDH>
void CompressedInspectorTask<RDH>::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "CompressedInspector run";

  /** check status **/
  if (mStatus) {
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  /** loop over inputs routes **/
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid()) {
      continue;
    }

    /** loop over input parts **/
    for (auto const& ref : iit) {

      const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;

      DecoderBaseT<RDH>::setDecoderBuffer(payloadIn);
      DecoderBaseT<RDH>::setDecoderBufferSize(payloadInSize);
      DecoderBaseT<RDH>::run();
    }
  }
}

template <typename RDH>
void CompressedInspectorTask<RDH>::headerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit)
{
  mHistos2D["crateBC"]->Fill(crateHeader->drmID, crateHeader->bunchID);
  mHistos2D["crateOrbit"]->Fill(crateHeader->drmID, crateOrbit->orbitID % 4096);

  for (int ibit = 0; ibit < 11; ++ibit) {
    if (crateHeader->slotPartMask & (1 << ibit)) {
      mHistos2D["slotPartMask"]->Fill(crateHeader->drmID, ibit + 2);
    }
  }
};

template <typename RDH>
void CompressedInspectorTask<RDH>::frameHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
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

template <typename RDH>
void CompressedInspectorTask<RDH>::trailerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                                                  const CrateTrailer_t* crateTrailer, const Diagnostic_t* diagnostics,
                                                  const Error_t* errors)
{
  mHistos2D["diagnostic"]->Fill(crateHeader->drmID, 0);
  for (int i = 0; i < crateTrailer->numberOfDiagnostics; ++i) {
    auto diagnostic = diagnostics + i;
    mHistos2D["diagnostic"]->Fill(crateHeader->drmID, diagnostic->slotID);
  }
  int nError = 0, nTest = 0;
  for (int i = 0; i < crateTrailer->numberOfErrors; ++i) {
    auto error = errors + i;
    if (error->undefined) {
      nTest++;
      mHistos2D["test"]->Fill(error->slotID + 0.5 * error->chain, error->tdcID);
    } else {
      nError++;
      mHistos2D["error"]->Fill(error->slotID + 0.5 * error->chain, error->tdcID);
      for (int ibit = 0; ibit < 15; ++ibit) {
        if (error->errorFlags & (1 << ibit)) {
          mHistos1D["errorBit"]->Fill(ibit);
        }
      }
    }
  }
  mHistos1D["Nerror"]->Fill(nError);
  mHistos1D["Ntest"]->Fill(nTest);
};

template class CompressedInspectorTask<o2::header::RAWDataHeaderV4>;
template class CompressedInspectorTask<o2::header::RAWDataHeaderV6>;

} // namespace tof
} // namespace o2
