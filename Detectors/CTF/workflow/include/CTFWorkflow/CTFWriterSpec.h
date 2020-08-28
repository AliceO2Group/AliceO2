// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CTFWriterSpec.h

#ifndef O2_CTFWRITER_SPEC
#define O2_CTFWRITER_SPEC

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "CommonUtils/StringUtils.h"
#include "rANS/rans.h"
#include <vector>
#include <array>
#include <TStopwatch.h>

namespace o2
{
namespace ctf
{

using DetID = o2::detectors::DetID;
using FTrans = o2::rans::FrequencyTable;

class CTFWriterSpec : public o2::framework::Task
{
 public:
  CTFWriterSpec() = delete;
  CTFWriterSpec(DetID::mask_t dm, uint64_t r = 0, bool doCTF = true, bool doDict = false, bool dictPerDet = false);
  ~CTFWriterSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  bool isPresent(DetID id) const { return mDets[id]; }

 private:
  template <typename C>
  void processDet(o2::framework::ProcessingContext& pc, DetID det, CTFHeader& header, TTree* tree);
  template <typename C>
  void storeDictionary(DetID det, CTFHeader& header);
  void storeDictionaries();
  void prepareDictionaryTreeAndFile(DetID det);
  void closeDictionaryTreeAndFile(CTFHeader& header);
  std::string dictionaryFileName(const std::string& detName = "");

  DetID::mask_t mDets; // detectors
  bool mWriteCTF = false;
  bool mCreateDict = false;
  bool mDictPerDetector = false;
  size_t mNTF = 0;
  int mSaveDictAfter = -1; // if positive and mWriteCTF==true, save dictionary after each mSaveDictAfter TFs processed
  uint64_t mRun = 0;

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

// process data of particular detector
template <typename C>
void CTFWriterSpec::processDet(o2::framework::ProcessingContext& pc, DetID det, CTFHeader& header, TTree* tree)
{
  if (!isPresent(det) || !pc.inputs().isValid(det.getName())) {
    return;
  }
  auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
  const auto ctfImage = C::getImage(ctfBuffer.data());
  ctfImage.print(o2::utils::concat_string(det.getName(), ": "));
  if (mWriteCTF) {
    ctfImage.appendToTree(*tree, det.getName());
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
}

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
  C::get(dictBlocks.data())->print(o2::utils::concat_string("Storing dictionary for ", det.getName(), ": "));
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

/// create a processor spec
framework::DataProcessorSpec getCTFWriterSpec(DetID::mask_t dets, uint64_t run, bool doCTF = true, bool doDict = false, bool dictPerDet = false);

} // namespace ctf
} // namespace o2

#endif /* O2_CTFWRITER_SPEC */
