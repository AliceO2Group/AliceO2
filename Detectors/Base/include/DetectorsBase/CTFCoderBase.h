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
#include "rANS/rans.h"
#include <filesystem>
#include "Framework/InitContext.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamRegistry.h"

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

  template <typename CTF>
  std::vector<char> readDictionaryFromFile(const std::string& dictPath, bool mayFail = false);

  template <typename CTF>
  void createCodersFromFile(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op, bool mayFail = false);

  template <typename S>
  void createCoder(OpType op, const o2::rans::RenormedFrequencyTable& renormedFrequencyTable, int slot)
  {
    if (renormedFrequencyTable.empty()) {
      LOG(warning) << "Empty dictionary provided for slot " << slot << ", " << (op == OpType::Encoder ? "encoding" : "decoding") << " will assume literal symbols only";
    }

    switch (op) {
      case OpType::Encoder:
        mCoders[slot].reset(new o2::rans::LiteralEncoder64<S>(renormedFrequencyTable));
        break;
      case OpType::Decoder:
        mCoders[slot].reset(new o2::rans::LiteralDecoder64<S>(renormedFrequencyTable));
        break;
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

  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);

  o2::utils::IRFrameSelector& getIRFramesSelector() { return mIRFrameSelector; }

 protected:
  std::string getPrefix() const { return o2::utils::Str::concat_string(mDet.getName(), "_CTF: "); }

  void checkDictVersion(const CTFDictHeader& h) const;

  std::vector<std::shared_ptr<void>> mCoders; // encoders/decoders
  DetID mDet;
  CTFDictHeader mExtHeader;      // external dictionary header
  o2::utils::IRFrameSelector mIRFrameSelector; // optional IR frames selector
  float mMemMarginFactor = 1.0f; // factor for memory allocation in EncodedBlocks
  bool mLoadDictFromCCDB{true};
  OpType mOpType; // Encoder or Decoder
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
  std::unique_ptr<TTree> tree((TTree*)fileDict->Get(std::string(o2::base::NameConf::CTFDICT).c_str()));
  std::unique_ptr<std::vector<char>> bv((std::vector<char>*)fileDict->GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), "std::vector<char>"));
  if (tree) {
    CTFHeader ctfHeader;
    if (!readFromTree(*tree.get(), "CTFHeader", ctfHeader) || !ctfHeader.detectors[mDet]) {
      std::string errstr = fmt::format("CTF dictionary file for detector {} is absent in the tree from file {}", mDet.getName(), dictPath);
      if (mayFail) {
        LOGP(info, "{}, will assume dictionary stored in CTF", errstr);
      } else {
        throw std::runtime_error(errstr);
      }
      return bufVec;
    }
    CTF::readFromTree(bufVec, *tree.get(), mDet.getName());
  } else if (bv) {
    bufVec.swap(*bv);
    if (bufVec.size()) {
      auto dictHeader = static_cast<const o2::ctf::CTFDictHeader&>(CTF::get(bufVec.data())->getHeader());
      if (dictHeader.det != mDet) {
        throw std::runtime_error(fmt::format("{} contains dictionary vector for {}, expected {}", dictPath, dictHeader.det.getName(), mDet.getName()));
      }
    }
  }
  if (bufVec.size()) {
    mExtHeader = static_cast<CTFDictHeader&>(CTF::get(bufVec.data())->getHeader());
    LOGP(debug, "Found {} in {}", mExtHeader.asString(), dictPath);
  } else {
    std::string errstr = fmt::format("CTF dictionary file for detector {} is empty", mDet.getName());
    if (mayFail) {
      LOGP(info, "{}, will assume dictionary stored in CTF", errstr);
    } else {
      throw std::runtime_error(errstr);
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
      createCoders(*dict, mOpType);
      mExtHeader = static_cast<const CTFDictHeader&>(CTF::get(dict->data())->getHeader());
      LOGP(info, "Loaded {} from CCDB", mExtHeader.asString());
    }
    mLoadDictFromCCDB = false; // we read the dictionary at most once!
  }
  return match;
}

} // namespace ctf
} // namespace o2

#endif
