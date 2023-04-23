// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EncoderImpl.h
/// @author Michael Lettrich
/// @brief  Defines the common operations for encoding data onto an rANS stream

#ifndef RANS_INTERNAL_ENCODE_ENCODERIMPL_H_
#define RANS_INTERNAL_ENCODE_ENCODERIMPL_H_

#include <cstdint>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

template <typename symbol_T, typename derived_T>
class EncoderImpl
{
 public:
  using stream_type = uint32_t;
  using state_type = uint64_t;
  using symbol_type = symbol_T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  [[nodiscard]] inline static constexpr size_type getNstreams() noexcept
  {
    return derived_T::getNstreams();
  };

  // Flushes the rANS encoder.
  template <typename Stream_IT>
  [[nodiscard]] inline Stream_IT flush(Stream_IT outputIter)
  {
    return static_cast<derived_T*>(this)->flush(outputIter);
  };

  template <typename Stream_IT>
  [[nodiscard]] inline Stream_IT putSymbols(Stream_IT outputIter, const symbol_type& encodeSymbols)
  {
    return static_cast<derived_T*>(this)->putSymbols(outputIter, encodeSymbols);
  };

  template <typename Stream_IT>
  [[nodiscard]] inline Stream_IT putSymbols(Stream_IT outputIter, const symbol_type& encodeSymbols, size_type nActiveStreams)
  {
    return static_cast<derived_T*>(this)->putSymbols(outputIter, encodeSymbols, nActiveStreams);
  };

  [[nodiscard]] inline static constexpr state_type getStreamingLowerBound() noexcept
  {
    return derived_T::getStreamingLowerBound();
  };

 protected:
  [[nodiscard]] inline static constexpr state_type getStreamOutTypeBits() noexcept
  {
    return utils::toBits<stream_type>();
  };

  EncoderImpl() = default;
  explicit EncoderImpl(size_t symbolTablePrecision) noexcept : mSymbolTablePrecision{symbolTablePrecision} {};

  size_type mSymbolTablePrecision{};
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_ENCODE_ENCODERIMPL_H_ */
