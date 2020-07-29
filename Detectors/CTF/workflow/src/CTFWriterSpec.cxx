// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CTFWriterSpec.cxx

#include <vector>
#include <TFile.h>
#include <TTree.h>

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "CommonUtils/StringUtils.h"
#include "CTFWorkflow/CTFWriterSpec.h"
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsTOF/CTF.h"

using namespace o2::framework;

namespace o2
{
namespace ctf
{

template <typename T>
void appendToTree(TTree& tree, const std::string brname, T& ptr)
{
  auto* br = tree.GetBranch(brname.c_str());
  auto* pptr = &ptr;
  if (br) {
    br->SetAddress(&pptr);
  } else {
    br = tree.Branch(brname.c_str(), &pptr);
  }
  br->Fill();
  br->ResetAddress();
}

void CTFWriterSpec::init(InitContext& ic)
{
}

void CTFWriterSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  auto tfOrb = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->firstTForbit;
  TFile flOut(o2::base::NameConf::getCTFFileName(tfOrb).c_str(), "recreate");
  TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");

  // create header
  CTFHeader header{mRun, tfOrb};

  DetID det;

  det = DetID::ITS;
  if (isPresent(det) && pc.inputs().isValid(det.getName())) {
    auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
    const auto ctfImage = o2::itsmft::CTF::getImage(ctfBuffer.data());
    LOG(INFO) << "CTF for " << det.getName();
    ctfImage.print();
    ctfImage.appendToTree(ctfTree, det.getName());
    header.detectors.set(det);
  }

  det = DetID::MFT;
  if (isPresent(det) && pc.inputs().isValid(det.getName())) {
    auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
    const auto ctfImage = o2::itsmft::CTF::getImage(ctfBuffer.data());
    LOG(INFO) << "CTF for " << det.getName();
    ctfImage.print();
    ctfImage.appendToTree(ctfTree, det.getName());
    header.detectors.set(det);
  }

  det = DetID::TPC;
  if (isPresent(det) && pc.inputs().isValid(det.getName())) {
    auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
    const auto ctfImage = o2::tpc::CTF::getImage(ctfBuffer.data());
    LOG(INFO) << "CTF for " << det.getName();
    ctfImage.print();
    ctfImage.appendToTree(ctfTree, det.getName());
    header.detectors.set(det);
  }

  det = DetID::FT0;
  if (isPresent(det) && pc.inputs().isValid(det.getName())) {
    auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
    const auto ctfImage = o2::ft0::CTF::getImage(ctfBuffer.data());
    LOG(INFO) << "CTF for " << det.getName();
    ctfImage.print();
    ctfImage.appendToTree(ctfTree, det.getName());
    header.detectors.set(det);
  }

  det = DetID::TOF;
  if (isPresent(det) && pc.inputs().isValid(det.getName())) {
    auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
    const auto ctfImage = o2::tof::CTF::getImage(ctfBuffer.data());
    LOG(INFO) << "CTF for " << det.getName();
    ctfImage.print();
    ctfImage.appendToTree(ctfTree, det.getName());
    header.detectors.set(det);
  }

  appendToTree(ctfTree, "CTFHeader", header);

  ctfTree.SetEntries(1);
  ctfTree.Write();
  flOut.Close();

  mTimer.Stop();
  LOG(INFO) << "Wrote " << flOut.GetName() << " with CTF{" << header << "} in " << mTimer.CpuTime() - cput << " s";
}

void CTFWriterSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "CTF writing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getCTFWriterSpec(DetID::mask_t dets, uint64_t run)
{
  std::vector<InputSpec> inputs;
  LOG(INFO) << "Det list:";
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (dets[id]) {
      inputs.emplace_back(DetID::getName(id), DetID::getDataOrigin(id), "CTFDATA", 0, Lifetime::Timeframe);
      LOG(INFO) << "Det " << DetID::getName(id) << " added";
    }
  }
  return DataProcessorSpec{
    "ctf-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<CTFWriterSpec>(dets, run)},
    Options{}};
}

} // namespace ctf
} // namespace o2
