// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCEntropyEncoder.h
/// @author Michael Lettrich
/// @since  Apr 30, 2020
/// @brief

#ifndef TPCENTROPYCODING_TPCENTROPYENCODER_H_
#define TPCENTROPYCODING_TPCENTROPYENCODER_H_

#include <memory>

#include <TTree.h>

#include "Framework/Logger.h"
#include "EncodedClusters.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "librans/rans.h"

namespace o2
{
namespace tpc
{

class TPCEntropyEncoder
{
 public:
  TPCEntropyEncoder() = delete;

  static std::unique_ptr<EncodedClusters> encode(const CompressedClusters& clusters);

  static void appendToTTree(TTree& tree, EncodedClusters& encodedClusters);

 private:
  template <typename source_T>
  static void compress(const std::string& name,           // name
                       const source_T* const sourceBegin, // begin of source message
                       const source_T* const sourceEnd,   // end of source message
                       uint8_t probabilityBits,           // encoding into
                       EncodedClusters& ec);              // structure to hold encoded data

  static size_t calculateMaxBufferSize(size_t num, size_t rangeBits, size_t sizeofStreamT);

  //rans default values
  const static inline size_t sProbabilityBits8Bit = 10;
  const static inline size_t sProbabilityBits16Bit = 22;
  const static inline size_t sProbabilityBits25Bit = 25;
};

template <typename source_T>
void TPCEntropyEncoder::compress(const std::string& name,           // name
                                 const source_T* const sourceBegin, // begin of source message
                                 const source_T* const sourceEnd,   // end of source message
                                 uint8_t probabilityBits,           // encoding into
                                 EncodedClusters& ec)               // structure to hold encoded data
{

  // find which array we are dealing with
  auto it = std::find(ec.NAMES.begin(), ec.NAMES.end(), name);
  assert(it != ec.NAMES.end());
  auto i = std::distance(ec.NAMES.begin(), it);

  // get references to the right data
  auto& dicts = ec.dicts;
  auto& buffers = ec.buffers;
  auto& md = *ec.metadata;

  // symbol statistics and encoding
  using stream_t = typename rans::Encoder64<source_T>::stream_t;
  rans::SymbolStatistics stats{sourceBegin, sourceEnd};
  stats.rescaleToNBits(probabilityBits);
  const auto buffSize = calculateMaxBufferSize(stats.getMessageLength(),
                                               stats.getAlphabetRangeBits(),
                                               sizeof(source_T));

  std::vector<stream_t> encoderBuffer(buffSize);
  const rans::Encoder64<source_T> encoder{stats, probabilityBits};
  const auto encodedMessageStart = encoder.process(encoderBuffer.begin(),
                                                   encoderBuffer.end(),
                                                   sourceBegin, sourceEnd);

  // write metadata
  md.emplace_back(EncodedClusters::Metadata{sizeof(uint64_t), sizeof(stream_t), probabilityBits, stats.getMinSymbol(), stats.getMaxSymbol()});

  //write dict
  dicts[i] = new std::vector<uint32_t>();
  for (const auto& item : stats) {
    dicts[i]->push_back(item.first);
  }

  // write buffer
  buffers[i] = new std::vector<uint32_t>(encodedMessageStart, encoderBuffer.end());

  LOG(INFO) << "finished entropy coding of " << name;
}

} // namespace tpc
} // namespace o2

#endif /* TPCENTROPYCODING_TPCENTROPYENCODER_H_ */
