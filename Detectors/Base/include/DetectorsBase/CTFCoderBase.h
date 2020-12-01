// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DetectorsCommonDataFormats/NameConf.h"
#include "rANS/rans.h"

namespace o2
{
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
  CTFCoderBase(int n, DetID det) : mCoders(n), mDet(det) {}

  std::unique_ptr<TFile> loadDictionaryTreeFile(const std::string& dictPath, bool mayFail = false);

  template <typename CTF>
  std::vector<char> readDictionaryFromFile(const std::string& dictPath, bool mayFail = false)
  {
    std::vector<char> bufVec;
    auto fileDict = loadDictionaryTreeFile(dictPath, mayFail);
    if (fileDict) {
      std::unique_ptr<TTree> tree((TTree*)fileDict->Get(std::string(o2::base::NameConf::CTFDICT).c_str()));
      CTF::readFromTree(bufVec, *tree.get(), mDet.getName());
    }
    return bufVec;
  }

  template <typename S>
  void createCoder(OpType op, const o2::rans::FrequencyTable& freq, uint8_t probabilityBits, int slot)
  {
    switch (op) {
      case OpType::Encoder:
        mCoders[slot].reset(new o2::rans::LiteralEncoder64<S>(freq, probabilityBits));
        break;
      case OpType::Decoder:
        mCoders[slot].reset(new o2::rans::LiteralDecoder64<S>(freq, probabilityBits));
        break;
    }
  }

  void clear()
  {
    for (auto c : mCoders) {
      c.reset();
    }
  }

 protected:
  std::string getPrefix() const { return o2::utils::concat_string(mDet.getName(), "_CTF: "); }

  std::vector<std::shared_ptr<void>> mCoders; // encoders/decoders
  DetID mDet;

  ClassDefNV(CTFCoderBase, 1);
};

} // namespace ctf
} // namespace o2

#endif
