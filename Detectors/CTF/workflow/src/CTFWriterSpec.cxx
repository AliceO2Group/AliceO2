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

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "CTFWorkflow/CTFWriterSpec.h"

#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "CommonUtils/StringUtils.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsTRD/CTF.h"
#include "DataFormatsHMP/CTF.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsFV0/CTF.h"
#include "DataFormatsFDD/CTF.h"
#include "DataFormatsTOF/CTF.h"
#include "DataFormatsMID/CTF.h"
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsEMCAL/CTF.h"
#include "DataFormatsPHOS/CTF.h"
#include "DataFormatsCPV/CTF.h"
#include "DataFormatsZDC/CTF.h"
#include "rANS/rans.h"
#include <vector>
#include <array>
#include <TStopwatch.h>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <filesystem>

using namespace o2::framework;

namespace o2
{
namespace ctf
{

template <typename T>
size_t appendToTree(TTree& tree, const std::string brname, T& ptr)
{
  size_t s = 0;
  auto* br = tree.GetBranch(brname.c_str());
  auto* pptr = &ptr;
  if (br) {
    br->SetAddress(&pptr);
  } else {
    br = tree.Branch(brname.c_str(), &pptr);
  }
  int res = br->Fill();
  if (res < 0) {
    throw std::runtime_error(fmt::format("Failed to fill CTF branch {}", brname));
  }
  s += res;
  br->ResetAddress();
  return s;
}

using DetID = o2::detectors::DetID;
using FTrans = o2::rans::FrequencyTable;

class CTFWriterSpec : public o2::framework::Task
{
 public:
  CTFWriterSpec() = delete;
  CTFWriterSpec(DetID::mask_t dm, uint64_t r = 0, bool doCTF = true, bool doDict = false, bool dictPerDet = false, size_t smn = 0, size_t szmx = 0);
  ~CTFWriterSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  bool isPresent(DetID id) const { return mDets[id]; }

 private:
  template <typename C>
  size_t processDet(o2::framework::ProcessingContext& pc, DetID det, CTFHeader& header, TTree* tree);
  template <typename C>
  void storeDictionary(DetID det, CTFHeader& header);
  void storeDictionaries();
  void prepareDictionaryTreeAndFile(DetID det);
  void closeDictionaryTreeAndFile(CTFHeader& header);
  std::string dictionaryFileName(const std::string& detName = "");
  void closeTFTreeAndFile();
  void prepareTFTreeAndFile(const o2::header::DataHeader* dh);
  size_t estimateCTFSize(ProcessingContext& pc);

  DetID::mask_t mDets; // detectors
  bool mWriteCTF = false;
  bool mCreateDict = false;
  bool mDictPerDetector = false;
  int mSaveDictAfter = -1; // if positive and mWriteCTF==true, save dictionary after each mSaveDictAfter TFs processed
  uint64_t mRun = 0;
  size_t mMinSize = 0;     // if > 0, accumulate CTFs in the same tree until the total size exceeds this minimum
  size_t mMaxSize = 0;     // if > MinSize, and accumulated size will exceed this value, stop accumulation (even if mMinSize is not reached)
  size_t mAccCTFSize = 0;  // so far accumulated size (if any)
  size_t mCurrCTFSize = 0; // size of currently processed CTF
  size_t mNCTF = 0;        // total number of CTFs written
  size_t mNAccCTF = 0;     // total number of CTFs accumulated in the current file
  size_t mNCTFFiles = 0;   // total number of CTF files written

  std::string mDictDir = "";
  std::string mCTFDir = "";

  std::unique_ptr<TFile> mCTFFileOut;
  std::unique_ptr<TTree> mCTFTreeOut;

  std::unique_ptr<TFile> mDictFileOut; // file to store dictionary
  std::unique_ptr<TTree> mDictTreeOut; // tree to store dictionary

  // For the external dictionary creation we accumulate for each detector the frequency tables of its each block
  // After accumulation over multiple TFs we store the dictionaries data in the standard CTF format of this detector,
  // i.e. EncodedBlock stored in a tree, BUT with dictionary data only added to each block.
  // The metadata of the block (min,max) will be used for the consistency check at the decoding
  std::array<std::vector<FTrans>, DetID::nDetectors> mFreqsAccumulation;
  std::array<std::vector<o2::ctf::Metadata>, DetID::nDetectors> mFreqsMetaData;
  std::array<std::shared_ptr<void>, DetID::nDetectors> mHeaders;

  TStopwatch mTimer;
};

//___________________________________________________________________
// process data of particular detector
template <typename C>
size_t CTFWriterSpec::processDet(o2::framework::ProcessingContext& pc, DetID det, CTFHeader& header, TTree* tree)
{
  size_t sz = 0;
  if (!isPresent(det) || !pc.inputs().isValid(det.getName())) {
    return sz;
  }
  auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
  const auto ctfImage = C::getImage(ctfBuffer.data());
  ctfImage.print(o2::utils::Str::concat_string(det.getName(), ": "));
  if (mWriteCTF) {
    sz += ctfImage.appendToTree(*tree, det.getName());
    header.detectors.set(det);
  }
  if (mCreateDict) {
    if (!mFreqsAccumulation[det].size()) {
      mFreqsAccumulation[det].resize(C::getNBlocks());
      mFreqsMetaData[det].resize(C::getNBlocks());
    }
    if (!mHeaders[det]) { // store 1st header
      mHeaders[det] = ctfImage.cloneHeader();
    }
    for (int ib = 0; ib < C::getNBlocks(); ib++) {
      const auto& bl = ctfImage.getBlock(ib);
      if (bl.getNDict()) {
        auto& freq = mFreqsAccumulation[det][ib];
        auto& mdSave = mFreqsMetaData[det][ib];
        const auto& md = ctfImage.getMetadata(ib);
        freq.addFrequencies(bl.getDict(), bl.getDict() + bl.getNDict(), md.min, md.max);
        mdSave = o2::ctf::Metadata{0, 0, md.coderType, md.streamSize, md.probabilityBits, md.opt, freq.getMinSymbol(), freq.getMaxSymbol(), (int)freq.size(), 0, 0};
      }
    }
  }
  return sz;
}

//___________________________________________________________________
// store dictionary of a particular detector
template <typename C>
void CTFWriterSpec::storeDictionary(DetID det, CTFHeader& header)
{
  if (!isPresent(det) || !mFreqsAccumulation[det].size()) {
    return;
  }
  prepareDictionaryTreeAndFile(det);
  // create vector whose data contains dictionary in CTF format (EncodedBlock)
  auto dictBlocks = C::createDictionaryBlocks(mFreqsAccumulation[det], mFreqsMetaData[det]);
  auto& h = C::get(dictBlocks.data())->getHeader();
  h = *reinterpret_cast<typename std::remove_reference<decltype(h)>::type*>(mHeaders[det].get());
  C::get(dictBlocks.data())->print(o2::utils::Str::concat_string("Storing dictionary for ", det.getName(), ": "));
  C::get(dictBlocks.data())->appendToTree(*mDictTreeOut.get(), det.getName()); // cast to EncodedBlock
  //  mFreqsAccumulation[det].clear();
  //  mFreqsMetaData[det].clear();
  if (mDictPerDetector) {
    header.detectors.reset();
  }
  header.detectors.set(det);
  if (mDictPerDetector) {
    closeDictionaryTreeAndFile(header);
  }
}

//___________________________________________________________________
CTFWriterSpec::CTFWriterSpec(DetID::mask_t dm, uint64_t r, bool doCTF, bool doDict, bool dictPerDet, size_t szmn, size_t szmx)
  : mDets(dm), mRun(r), mWriteCTF(doCTF), mCreateDict(doDict), mDictPerDetector(dictPerDet), mMinSize(szmn), mMaxSize(szmx)
{
  mTimer.Stop();
  mTimer.Reset();

  if (doDict) { // make sure that there is no local dictonary
    for (int id = 0; id < DetID::nDetectors; id++) {
      DetID det(id);
      if (isPresent(det)) {
        auto dictName = dictionaryFileName(det.getName());
        if (std::filesystem::exists(dictName)) {
          throw std::runtime_error(o2::utils::Str::concat_string("CTF dictionary creation is requested but ", dictName, " already exists, remove it!"));
        }
        if (!mDictPerDetector) {
          break; // no point in checking further
        }
      }
    }
  }
}

//___________________________________________________________________
void CTFWriterSpec::init(InitContext& ic)
{
  mSaveDictAfter = ic.options().get<int>("save-dict-after");
  mDictDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("ctf-dict-dir"));
  mCTFDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("output-dir"));
  if (mWriteCTF) {
    if (mMinSize > 0) {
      LOG(INFO) << "Multiple CTFs will be accumulated in the tree/file until its size exceeds " << mMinSize << " bytes";
      if (mMaxSize > mMinSize) {
        LOG(INFO) << "but does not exceed " << mMaxSize << " bytes";
      }
    }
  }
}

//___________________________________________________________________
size_t CTFWriterSpec::estimateCTFSize(ProcessingContext& pc)
{
  size_t s = 0;
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    DetID det(id);
    if (!isPresent(det) || !pc.inputs().isValid(det.getName())) {
      continue;
    }
    s += pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName()).size();
  }
  return s;
}

//___________________________________________________________________
void CTFWriterSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0));

  mCurrCTFSize = estimateCTFSize(pc);
  if (mWriteCTF) {
    prepareTFTreeAndFile(dh);
  }

  // create header
  CTFHeader header{mRun, dh->firstTForbit};
  size_t szCTF = 0;
  szCTF += processDet<o2::itsmft::CTF>(pc, DetID::ITS, header, mCTFTreeOut.get());
  szCTF += processDet<o2::itsmft::CTF>(pc, DetID::MFT, header, mCTFTreeOut.get());
  szCTF += processDet<o2::tpc::CTF>(pc, DetID::TPC, header, mCTFTreeOut.get());
  szCTF += processDet<o2::trd::CTF>(pc, DetID::TRD, header, mCTFTreeOut.get());
  szCTF += processDet<o2::tof::CTF>(pc, DetID::TOF, header, mCTFTreeOut.get());
  szCTF += processDet<o2::ft0::CTF>(pc, DetID::FT0, header, mCTFTreeOut.get());
  szCTF += processDet<o2::fv0::CTF>(pc, DetID::FV0, header, mCTFTreeOut.get());
  szCTF += processDet<o2::fdd::CTF>(pc, DetID::FDD, header, mCTFTreeOut.get());
  szCTF += processDet<o2::mid::CTF>(pc, DetID::MID, header, mCTFTreeOut.get());
  szCTF += processDet<o2::mch::CTF>(pc, DetID::MCH, header, mCTFTreeOut.get());
  szCTF += processDet<o2::emcal::CTF>(pc, DetID::EMC, header, mCTFTreeOut.get());
  szCTF += processDet<o2::phos::CTF>(pc, DetID::PHS, header, mCTFTreeOut.get());
  szCTF += processDet<o2::cpv::CTF>(pc, DetID::CPV, header, mCTFTreeOut.get());
  szCTF += processDet<o2::zdc::CTF>(pc, DetID::ZDC, header, mCTFTreeOut.get());
  szCTF += processDet<o2::hmpid::CTF>(pc, DetID::HMP, header, mCTFTreeOut.get());

  mTimer.Stop();

  if (mWriteCTF) {
    szCTF += appendToTree(*mCTFTreeOut.get(), "CTFHeader", header);
    mAccCTFSize += szCTF;
    mCTFTreeOut->SetEntries(++mNAccCTF);
    LOG(INFO) << "TF#" << mNCTF << ": wrote CTF{" << header << "} of size " << szCTF << " to " << mCTFFileOut->GetName() << " in " << mTimer.CpuTime() - cput << " s";
    if (mNAccCTF > 1) {
      LOG(INFO) << "Current CTF tree has " << mNAccCTF << " entries with total size of " << mAccCTFSize << " bytes";
    }
  } else {
    LOG(INFO) << "TF#" << mNCTF << " CTF writing is disabled, size was " << szCTF << " bytes";
  }

  if (mWriteCTF && mAccCTFSize >= mMinSize) {
    closeTFTreeAndFile();
  }

  mNCTF++;
  if (mCreateDict && mSaveDictAfter > 0 && (mNCTF % mSaveDictAfter) == 0) {
    storeDictionaries();
  }
}

//___________________________________________________________________
void CTFWriterSpec::endOfStream(EndOfStreamContext& ec)
{

  if (mCreateDict) {
    storeDictionaries();
  }
  if (mWriteCTF) {
    closeTFTreeAndFile();
  }
  LOGF(INFO, "CTF writing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

//___________________________________________________________________
void CTFWriterSpec::prepareTFTreeAndFile(const o2::header::DataHeader* dh)
{
  if (!mWriteCTF) {
    return;
  }
  bool needToOpen = false;
  if (!mCTFTreeOut) {
    needToOpen = true;
  } else {
    if ((mAccCTFSize >= mMinSize) ||                                                         // min size exceeded, may close the file
        (mAccCTFSize && mMaxSize > mMinSize && ((mAccCTFSize + mCurrCTFSize) > mMaxSize))) { // this is not the 1st CTF in the file and the new size will exceed allowed max
      needToOpen = true;
    } else {
      LOGP(INFO, "Will add new CTF of estimated size {} to existing file of size {}", mCurrCTFSize, mAccCTFSize);
    }
  }
  if (needToOpen) {
    closeTFTreeAndFile();
    mCTFFileOut.reset(TFile::Open(o2::utils::Str::concat_string(mCTFDir, o2::base::NameConf::getCTFFileName(dh->runNumber, dh->firstTForbit, dh->tfCounter)).c_str(), "recreate"));
    mCTFTreeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    mNCTFFiles++;
  }
}

//___________________________________________________________________
void CTFWriterSpec::closeTFTreeAndFile()
{
  if (mCTFTreeOut) {
    mCTFTreeOut->Write();
    mCTFTreeOut.reset();
    mCTFFileOut->Close();
    mCTFFileOut.reset();
    mNAccCTF = 0;
  }
  mAccCTFSize = 0;
}

//___________________________________________________________________
void CTFWriterSpec::prepareDictionaryTreeAndFile(DetID det)
{
  if (mDictPerDetector) {
    if (mDictTreeOut) {
      mDictTreeOut->SetEntries(1);
      mDictTreeOut->Write();
      mDictTreeOut.reset();
      mDictFileOut.reset();
    }
  }
  if (!mDictTreeOut) {
    mDictFileOut.reset(TFile::Open(dictionaryFileName(det.getName()).c_str(), "recreate"));
    mDictTreeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFDICT).c_str(), "O2 CTF dictionary");
  }
}

//___________________________________________________________________
std::string CTFWriterSpec::dictionaryFileName(const std::string& detName)
{
  if (mDictPerDetector) {
    if (detName.empty()) {
      throw std::runtime_error("Per-detector dictionary files are requested but detector name is not provided");
    }
    return o2::utils::Str::concat_string(mDictDir, detName, '_', o2::base::NameConf::CTFDICT, ".root");
  } else {
    return o2::utils::Str::concat_string(mDictDir, o2::base::NameConf::CTFDICT, ".root");
  }
}

//___________________________________________________________________
void CTFWriterSpec::storeDictionaries()
{
  CTFHeader header{mRun, uint32_t(mNCTF)};
  storeDictionary<o2::itsmft::CTF>(DetID::ITS, header);
  storeDictionary<o2::itsmft::CTF>(DetID::MFT, header);
  storeDictionary<o2::tpc::CTF>(DetID::TPC, header);
  storeDictionary<o2::trd::CTF>(DetID::TRD, header);
  storeDictionary<o2::tof::CTF>(DetID::TOF, header);
  storeDictionary<o2::ft0::CTF>(DetID::FT0, header);
  storeDictionary<o2::fv0::CTF>(DetID::FV0, header);
  storeDictionary<o2::fdd::CTF>(DetID::FDD, header);
  storeDictionary<o2::mid::CTF>(DetID::MID, header);
  storeDictionary<o2::mch::CTF>(DetID::MCH, header);
  storeDictionary<o2::emcal::CTF>(DetID::EMC, header);
  storeDictionary<o2::phos::CTF>(DetID::PHS, header);
  storeDictionary<o2::cpv::CTF>(DetID::CPV, header);
  storeDictionary<o2::zdc::CTF>(DetID::ZDC, header);
  storeDictionary<o2::hmpid::CTF>(DetID::HMP, header);

  // close remnants
  if (mDictTreeOut) {
    closeDictionaryTreeAndFile(header);
  }
  LOG(INFO) << "Saved CTF dictionary after " << mNCTF << " TFs processed";
}

//___________________________________________________________________
void CTFWriterSpec::closeDictionaryTreeAndFile(CTFHeader& header)
{
  if (mDictTreeOut) {
    appendToTree(*mDictTreeOut.get(), "CTFHeader", header);
    mDictTreeOut->SetEntries(1);
    mDictTreeOut->Write(mDictTreeOut->GetName(), TObject::kSingleKey);
    mDictTreeOut.reset();
    mDictFileOut.reset();
  }
}

//___________________________________________________________________
DataProcessorSpec getCTFWriterSpec(DetID::mask_t dets, uint64_t run, bool doCTF, bool doDict, bool dictPerDet, size_t szmn, size_t szmx)
{
  std::vector<InputSpec> inputs;
  LOG(INFO) << "Detectors list:";
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
    AlgorithmSpec{adaptFromTask<CTFWriterSpec>(dets, run, doCTF, doDict, dictPerDet, szmn, szmx)},
    Options{{"save-dict-after", VariantType::Int, -1, {"In dictionary generation mode save it dictionary after certain number of TFs processed"}},
            {"ctf-dict-dir", VariantType::String, "none", {"CTF dictionary directory"}},
            {"output-dir", VariantType::String, "none", {"CTF output directory"}}}};
}

} // namespace ctf
} // namespace o2
