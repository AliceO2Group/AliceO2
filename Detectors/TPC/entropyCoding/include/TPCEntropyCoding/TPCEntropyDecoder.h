// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCEntropyDecoder.h
/// @author Michael Lettrich
/// @since  Apr 30, 2020
/// @brief

#ifndef TPCENTROPYCODING_TPCENTROPYDECODER_H_
#define TPCENTROPYCODING_TPCENTROPYDECODER_H_

#include <TTree.h>
#include <TFile.h>

#include "Framework/Logger.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "TPCEntropyCoding/EncodedClusters.h"
#include "librans/rans.h"

#include <iostream>
#include <string>
#include <cstring>

namespace o2
{
namespace tpc
{
class TPCEntropyDecoder
{
 public:
  TPCEntropyDecoder() = delete;

  static std::unique_ptr<EncodedClusters> fromTree(TTree& tree);

  static std::unique_ptr<CompressedClusters> initCompressedClusters(const EncodedClusters& ec);

  template <typename source_T>
  static std::unique_ptr<std::vector<source_T>> decodeEntry(source_T** destPPtr,
                                                            const std::string& name,
                                                            const EncodedClusters& ec,
                                                            size_t msgLength);
};

template <typename source_T>
std::unique_ptr<std::vector<source_T>> TPCEntropyDecoder::decodeEntry(source_T** destPPtr, const std::string& name, const EncodedClusters& ec, size_t msgLength)
{
  // find which array we are dealing with
  auto it = std::find(ec.NAMES.begin(), ec.NAMES.end(), name);
  assert(it != ec.NAMES.end());
  auto i = std::distance(ec.NAMES.begin(), it);

  // get references to the right data
  auto& dict = *ec.dicts[i];
  auto& buffer = *ec.buffers[i];
  auto& md = ec.metadata->at(i);

  // allocate output buffer
  auto outBuffer = std::make_unique<std::vector<source_T>>(msgLength);

  // decode
  rans::SymbolStatistics stats(dict.begin(), dict.end(), md.min, md.max, msgLength);
  rans::Decoder64<source_T> decoder(stats, md.probabilityBits);
  decoder.process(outBuffer->begin(), buffer.begin(), msgLength);

  LOG(INFO) << "finished decoding entry " << name;
  //return
  *destPPtr = outBuffer->data();
  return std::move(outBuffer);
}

} // namespace tpc
} // namespace o2

#endif /* TPCENTROPYCODING_TPCENTROPYDECODER_H_ */
