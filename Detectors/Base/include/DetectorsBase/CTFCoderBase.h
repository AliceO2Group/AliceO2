// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CTFCoderBase.h
/// \brief Declarations for CTFCoderBase class (support of external dictionaries)
/// \author ruben.shahoyan@cern.ch

#ifndef _ALICEO2_CTFCODER_BASE_H_
#define _ALICEO2_CTFCODER_BASE_H_

#include <memory>
#include <TFile.h>
#include <TTree.h>
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/IRFrameSelector.h"
#include "DetectorsCommonDataFormats/CTFDictHeader.h"
#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "DetectorsCommonDataFormats/CTFIOSize.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "DetectorsCommonDataFormats/ANSHeader.h"
#include "rANS/factory.h"
#include "rANS/compat.h"
#include "rANS/histogram.h"
#include <filesystem>
#include "Framework/InitContext.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamRegistry.h"
#include <any>

namespace o2
{
namespace framework
{
class ProcessingContext;
}
namespace ctf
{

/// this is a base class for particular detector CTF coder/decoder, provides common
/// interface to create external entropy encoders/decoders

using DetID = o2::detectors::DetID;

class CTFCoderBase
{

 public:
  enum class OpType : int { Encoder,
                            Decoder };

  CTFCoderBase() = delete;
  CTFCoderBase(int n, DetID det, float memFactor = 1.f) : mCoders(n), mDet(det), mMemMarginFactor(memFactor > 1.f ? memFactor : 1.f) {}
  CTFCoderBase(OpType op, int n, DetID det, float memFactor = 1.f) : mOpType(op), mCoders(n), mDet(det), mMemMarginFactor(memFactor > 1.f ? memFactor : 1.f) {}
  virtual ~CTFCoderBase() = default;

  virtual void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) = 0;

  // detector coder need to redefine this method if uses no default version, see comment in the cxx file
  virtual void assignDictVersion(CTFDictHeader& h) const;

  template <typename SPAN>
  void setSelectedIRFrames(const SPAN& sp)
  {
    reportIRFrames();
    mIRFrameSelector.setSelectedIRFrames(sp, mIRFrameSelMarginBwd, mIRFrameSelMarginFwd, mIRFrameSelShift, true);
  }

  template <typename CTF>
  std::vector<char> readDictionaryFromFile(const std::string& dictPath, bool mayFail = false);

  template <typename CTF>
  void createCodersFromFile(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op, bool mayFail = false);

  template <typename S>
  void createCoder(OpType op, const o2::rans::RenormedDenseHistogram<S>& renormedHistogram, int slot)
  {
    LOG_IF(warning, renormedHistogram.empty()) << fmt::format("Empty dictionary provided for slot {}, {} will assume literal symbols only", slot, (op == OpType::Encoder ? "encoding" : "decoding"));

    if (mANSVersion == ANSVersionCompat) {
      switch (op) {
        case OpType::Encoder:
          mCoders[slot] = std::make_any<rans::compat::encoder_type<S>>(rans::compat::makeEncoder::fromRenormed(renormedHistogram));
          break;
        case OpType::Decoder:
          mCoders[slot] = std::make_any<rans::compat::decoder_type<S>>(rans::compat::makeDecoder::fromRenormed(renormedHistogram));
          break;
      }
    } else if (mANSVersion == ANSVersion1) {
      switch (op) {
        case OpType::Encoder:
          mCoders[slot] = std::make_any<rans::denseEncoder_type<S>>(rans::makeDenseEncoder<>::fromRenormed(renormedHistogram));
          break;
        case OpType::Decoder:
          mCoders[slot] = std::make_any<rans::defaultDecoder_type<S>>(rans::makeDecoder<>::fromRenormed(renormedHistogram));
          break;
      }
    } else {
      throw std::runtime_error("unsupported ANS version");
    }
  }

  void clear()
  {
    for (auto c : mCoders) {
      c.reset();
    }
  }

  void setMemMarginFactor(float v) { mMemMarginFactor = v > 1.f ? v : 1.f; }
  float getMemMarginFactor() const { return mMemMarginFactor; }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

  const CTFDictHeader& getExtDictHeader() const { return mExtHeader; }

  template <typename T>
  static bool readFromTree(TTree& tree, const std::string brname, T& dest, int ev = 0);

  // these are the helper methods for the parent encoding/decoding task
  template <typename CTF>
  void init(o2::framework::InitContext& ic);

  template <typename CTF, typename BUF>
  size_t finaliseCTFOutput(BUF& buffer);

  template <typename CTF>
  bool finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj);

  void updateTimeDependentParams(o2::framework::ProcessingContext& pc, bool askTree = false);

  o2::utils::IRFrameSelector& getIRFramesSelector() { return mIRFrameSelector; }
  size_t getIRFrameSelMarginBwd() const { return mIRFrameSelMarginBwd; }
  size_t getIRFrameSelMarginFwd() const { return mIRFrameSelMarginFwd; }
  long getIRFrameSelShift() const { return mIRFrameSelShift; }

  inline const ctf::ANSHeader& getANSVersion() const noexcept { return mANSVersion; };
  inline ctf::ANSHeader& getANSVersion() { return const_cast<ctf::ANSHeader&>(const_cast<const CTFCoderBase&>(*this).getANSVersion()); };
  inline void setANSVersion(const ctf::ANSHeader& ansVersion) noexcept { mANSVersion = ansVersion; };
  void setBCShift(int64_t n) { mBCShift = n; }
  void setFirstTFOrbit(uint32_t n) { mFirstTFOrbit = n; }
  auto getBCShift() const { return mBCShift; }
  auto getFirstTFOrbit() const { return mFirstTFOrbit; }
  void setSupportBCShifts(bool v = true) { mSupportBCShifts = v; }
  bool getSupportBCShifts() const { return mSupportBCShifts; }

  void setDictBinding(const std::string& s) { mDictBinding = s; }
  const std::string& getDictBinding() const { return mDictBinding; }

  void setTrigOffsBinding(const std::string& s) { mTrigOffsBinding = s; }
  const std::string& getTrigOffsBinding() const { return mTrigOffsBinding; }

  const DetID getDet() const { return mDet; }

 protected:
  void reportIRFrames();
  std::string getPrefix() const { return o2::utils::Str::concat_string(mDet.getName(), "_CTF: "); }
  void checkDictVersion(const CTFDictHeader& h) const;
  bool isTreeDictionary(const void* buff) const;
  bool canApplyBCShift(const o2::InteractionRecord& ir, long shift) const
  {
    auto diff = ir.differenceInBC({0, mFirstTFOrbit});
    return diff < 0 ? true : diff >= shift;
  }
  bool canApplyBCShift(const o2::InteractionRecord& ir) const { return canApplyBCShift(ir, mBCShift); }

  template <typename source_IT>
  [[nodiscard]] size_t estimateBufferSize(size_t slot, source_IT samplesBegin, source_IT samplesEnd);

  template <typename source_T>
  size_t estimateBufferSize(size_t slot, size_t nSamples);

  template <typename source_T>
  [[nodiscard]] size_t estimateBufferSize(size_t slot, const std::vector<source_T>& samples)
  {
    return estimateBufferSize(slot, samples.begin(), samples.end());
  }

  template <typename CTF>
  std::vector<char> loadDictionaryFromTree(TTree* tree);
  std::vector<std::any> mCoders; // encoders/decoders
  DetID mDet;
  std::string mDictBinding{"ctfdict"};
  std::string mTrigOffsBinding{"trigoffset"};
  CTFDictHeader mExtHeader;                    // external dictionary header
  o2::utils::IRFrameSelector mIRFrameSelector; // optional IR frames selector
  float mMemMarginFactor = 1.0f;               // factor for memory allocation in EncodedBlocks
  bool mLoadDictFromCCDB{true};
  bool mSupportBCShifts{false};
  OpType mOpType;                                    // Encoder or Decoder
  ctf::ANSHeader mANSVersion{ctf::ANSVersionCompat}; // Version of the ANSEncoder/Decoder
  int64_t mBCShift = 0;                              // shift to apply to decoded IR (i.e. CTP offset if was not corrected on raw data decoding level)
  uint32_t mFirstTFOrbit = 0;
  size_t mIRFrameSelMarginBwd = 0; // margin in BC to add to the IRFrame lower boundary when selection is requested
  size_t mIRFrameSelMarginFwd = 0; // margin in BC to add to the IRFrame upper boundary when selection is requested
  long mIRFrameSelShift = 0;       // Global shift of the IRFrames, to account for e.g. detector latency
  int mVerbosity = 0;
};

///________________________________
template <typename T>
bool CTFCoderBase::readFromTree(TTree& tree, const std::string brname, T& dest, int ev)
{
  auto* br = tree.GetBranch(brname.c_str());
  if (br && br->GetEntries() > ev) {
    auto* ptr = &dest;
    br->SetAddress(&ptr);
    br->GetEntry(ev);
    br->ResetAddress();
    return true;
  }
  return false;
}

///________________________________
template <typename CTF>
void CTFCoderBase::createCodersFromFile(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op, bool mayFail)
{
  auto buff = readDictionaryFromFile<CTF>(dictPath, mayFail);
  if (!buff.size()) {
    if (mayFail) {
      return;
    }
    throw std::runtime_error("Failed to create CTF dictionaty");
  }
  createCoders(buff, op);
}

///________________________________
template <typename CTF>
std::vector<char> CTFCoderBase::loadDictionaryFromTree(TTree* tree)
{
  std::vector<char> bufVec;
  CTFHeader ctfHeader;
  if (readFromTree(*tree, "CTFHeader", ctfHeader) && ctfHeader.detectors[mDet]) {
    CTF::readFromTree(bufVec, *tree, mDet.getName());
  }
  return bufVec;
}

///________________________________
template <typename CTF>
std::vector<char> CTFCoderBase::readDictionaryFromFile(const std::string& dictPath, bool mayFail)
{
  std::vector<char> bufVec;
  std::unique_ptr<TFile> fileDict;
  if (std::filesystem::exists(dictPath)) {
    fileDict.reset(TFile::Open(dictPath.c_str()));
  }
  if (!fileDict || fileDict->IsZombie()) {
    std::string errstr = fmt::format("CTF dictionary file {} for detector {} is absent", dictPath, mDet.getName());
    if (mayFail) {
      LOGP(info, "{}, will assume dictionary stored in CTF", errstr);
    } else {
      throw std::runtime_error(errstr);
    }
    return bufVec;
  }
  std::unique_ptr<TTree> tree;
  std::unique_ptr<std::vector<char>> bv((std::vector<char>*)fileDict->GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), "std::vector<char>"));
  tree.reset((TTree*)fileDict->GetObjectChecked(o2::base::NameConf::CTFDICT.data(), "TTree"));
  if (!tree) {
    tree.reset((TTree*)fileDict->GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), "TTree"));
  }
  if (tree) {
    auto v = loadDictionaryFromTree<CTF>(tree.get());
    bufVec.swap(v);
  } else if (bv) {
    bufVec.swap(*bv);
    if (bufVec.size()) {
      auto dictHeader = static_cast<const o2::ctf::CTFDictHeader&>(CTF::get(bufVec.data())->getHeader());
      if (dictHeader.det != mDet) {
        bufVec.clear();
        LOGP(error, "{} contains dictionary vector for {}, expected {}", dictPath, dictHeader.det.getName(), mDet.getName());
      }
    }
  }
  if (bufVec.size()) {
    mExtHeader = static_cast<CTFDictHeader&>(CTF::get(bufVec.data())->getHeader());
    LOGP(debug, "Found {} in {}", mExtHeader.asString(), dictPath);
  } else {
    if (mayFail) {
      LOGP(info, "{}, will assume dictionary stored in CTF", mDet.getName());
    } else {
      LOGP(fatal, "CTF dictionary file for detector {} is empty", mDet.getName());
    }
  }
  return bufVec;
}

///________________________________
template <typename CTF>
void CTFCoderBase::init(o2::framework::InitContext& ic)
{
  if (ic.options().hasOption("mem-factor")) {
    setMemMarginFactor(ic.options().get<float>("mem-factor"));
  }
  if (ic.options().hasOption("irframe-margin-bwd")) {
    mIRFrameSelMarginBwd = ic.options().get<uint32_t>("irframe-margin-bwd");
  }
  if (ic.options().hasOption("irframe-margin-fwd")) {
    mIRFrameSelMarginFwd = ic.options().get<uint32_t>("irframe-margin-fwd");
  }
  if (ic.options().hasOption("irframe-shift")) {
    mIRFrameSelShift = (long)ic.options().get<int32_t>("irframe-shift");
  }
  if (ic.options().hasOption("ans-version")) {
    if (ic.options().isSet("ans-version")) {
      const std::string ansVersionString = ic.options().get<std::string>("ans-version");
      if (!ansVersionString.empty()) {
        mANSVersion = ansVersionFromString(ansVersionString);
        LOGP(info, "parsing ansVersionString {} into {}", ansVersionString, static_cast<std::string>(mANSVersion));
        if (mANSVersion == ANSVersionUnspecified) {
          throw std::invalid_argument(fmt::format("Invalid ANS Version {}", ansVersionString));
        }
      }
    }
  }
  auto dict = ic.options().get<std::string>("ctf-dict");
  if (dict.empty() || dict == "ccdb") { // load from CCDB
    mLoadDictFromCCDB = true;
  } else {
    if (dict != "none") { // none means per-CTF dictionary will created on the fly
      createCodersFromFile<CTF>(dict, mOpType);
      LOGP(info, "Loaded {} from {}", mExtHeader.asString(), dict);
    } else {
      LOGP(info, "Internal per-TF CTF Dict will be created");
    }
    mLoadDictFromCCDB = false; // don't try to load from CCDB
  }
}

///________________________________
template <typename CTF, typename BUF>
size_t CTFCoderBase::finaliseCTFOutput(BUF& buffer)
{
  auto eeb = CTF::get(buffer.data()); // cast to container pointer
  eeb->compactify();                  // eliminate unnecessary padding
  buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  // eeb->print();
  return eeb->size();
}

///________________________________
template <typename CTF>
bool CTFCoderBase::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  bool match = false;
  if (mLoadDictFromCCDB && (match = (matcher == o2::framework::ConcreteDataMatcher(mDet.getDataOrigin(), "CTFDICT", 0)))) {
    const auto* dict = (std::vector<char>*)obj;
    if (dict->empty()) {
      LOGP(info, "Empty dictionary object fetched from CCDB, internal per-TF CTF Dict will be created");
    } else {
      std::vector<char> bufVec;
      if (isTreeDictionary(obj)) {
        auto v = loadDictionaryFromTree<CTF>(reinterpret_cast<TTree*>(obj));
        bufVec.swap(v);
        dict = &bufVec;
      }
      createCoders(*dict, mOpType);
      mExtHeader = static_cast<const CTFDictHeader&>(CTF::get(dict->data())->getHeader());
      LOGP(info, "Loaded {} from CCDB", mExtHeader.asString());
    }
    mLoadDictFromCCDB = false; // we read the dictionary at most once!
  } else if ((match = (matcher == o2::framework::ConcreteDataMatcher("CTP", "Trig_Offset", 0)))) {
    const auto& trigOffsParam = o2::ctp::TriggerOffsetsParam::Instance();
    auto bcshift = trigOffsParam.customOffset[mDet.getID()];
    if (bcshift) {
      if (mSupportBCShifts) {
        LOGP(info, "Decoded IRs will be augmented by {} BCs, discarded if become prior to 1st orbit", bcshift);
        setBCShift(-bcshift); // offset is subtracted
      } else {
        LOGP(alarm, "Decoding with {} BCs shift is requested, but the {} does not support this operation, ignoring request", bcshift, mDet.getName());
      }
    }
  }
  return match;
}

template <typename IT>
[[nodiscard]] inline size_t CTFCoderBase::estimateBufferSize(size_t slot, IT samplesBegin, IT samplesEnd)
{
  using source_type = typename std::iterator_traits<IT>::value_type;
  const size_t nSamples = std::distance(samplesBegin, samplesEnd);
  return estimateBufferSize<source_type>(slot, nSamples);
};

template <typename source_T>
[[nodiscard]] inline size_t CTFCoderBase::estimateBufferSize(size_t slot, size_t nSamples)
{

  std::any& coder = mCoders[slot];
  if (coder.has_value()) {
    const size_t alphabetRangeBits = [this, &coder]() {
      if (mANSVersion == ANSVersionCompat) {
        const auto& encoder = std::any_cast<const rans::compat::encoder_type<source_T>&>(coder);
        auto view = rans::trim(rans::makeHistogramView(encoder.getSymbolTable()));
        return rans::utils::getRangeBits(view.getMin(), view.getMax());
      } else if (mANSVersion == ANSVersion1) {
        const auto& encoder = std::any_cast<const rans::denseEncoder_type<source_T>&>(coder);
        auto view = rans::trim(rans::makeHistogramView(encoder.getSymbolTable()));
        return rans::utils::getRangeBits(view.getMin(), view.getMax());
      } else {
        throw std::runtime_error("unsupported ANS version");
      }
    }();
    return rans::compat::calculateMaxBufferSizeB(nSamples, alphabetRangeBits);
  } else {
    return nSamples * sizeof(source_T);
  }
};

} // namespace ctf
} // namespace o2

#endif
