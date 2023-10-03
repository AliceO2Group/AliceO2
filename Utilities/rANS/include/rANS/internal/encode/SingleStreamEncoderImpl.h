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

/// @file   SingleStreamEncoderImpl.h
/// @author Michael Lettrich
/// @brief  rANS encoding operations based on ryg's fast algorithm and a naive rANS implementation for all 64Bit CPUs.

#ifndef RANS_INTERNAL_ENCODE_SINGLESTREAMENCODERIMPL_H_
#define RANS_INTERNAL_ENCODE_SINGLESTREAMENCODERIMPL_H_

#include <cassert>
#include <cstdint>
#include <tuple>

#include "rANS/internal/common/defaults.h"
#include "rANS/internal/common/utils.h"
#include "rANS/internal/encode/EncoderImpl.h"
#include "rANS/internal/containers/Symbol.h"

namespace o2::rans::internal
{

template <size_t streamingLowerBound_V, typename symbol_T, typename derived_T>
class SingleStreamEncoderImplBase : public EncoderImpl<symbol_T,
                                                       SingleStreamEncoderImplBase<streamingLowerBound_V, symbol_T, derived_T>>
{
  using base_type = EncoderImpl<symbol_T, SingleStreamEncoderImplBase<streamingLowerBound_V, symbol_T, derived_T>>;

 public:
  using stream_type = typename base_type::stream_type;
  using state_type = typename base_type::state_type;
  using symbol_type = typename base_type::symbol_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;

  [[nodiscard]] inline static constexpr size_type getNstreams() noexcept { return 1ull; };

  template <typename stream_IT>
  [[nodiscard]] inline stream_IT flush(stream_IT outputIter)
  {
    static_assert(base_type::getStreamOutTypeBits() == 32);
    *(outputIter++) = static_cast<stream_type>(mState >> 32);
    *(outputIter++) = static_cast<stream_type>(mState);

    mState = 0;
    return outputIter;
  };

  template <typename Stream_IT>
  [[nodiscard]] inline Stream_IT putSymbols(Stream_IT outputIter, const symbol_type& encodeSymbols)
  {
    return static_cast<derived_T*>(this)->putSymbols(outputIter, encodeSymbols);
  };

  template <typename Stream_IT>
  [[nodiscard]] inline Stream_IT putSymbols(Stream_IT outputIter, const symbol_type& encodeSymbols, size_type nActiveStreams)
  {
    assert(nActiveStreams == 1);
    return putSymbols(outputIter, encodeSymbols);
  };

  [[nodiscard]] inline static constexpr state_type getStreamingLowerBound() noexcept { return static_cast<state_type>(utils::pow2(streamingLowerBound_V)); };

 protected:
  state_type mState{getStreamingLowerBound()};

  template <typename stream_IT>
  [[nodiscard]] inline std::tuple<state_type, stream_IT> renorm(state_type state, stream_IT outputIter, typename std::remove_pointer_t<typename std::remove_cv_t<symbol_type>>::value_type frequency)
  {
    state_type maxState = ((base_type::getStreamingLowerBound() >> this->mSymbolTablePrecision) << base_type::getStreamOutTypeBits()) * frequency;
    if (state >= maxState) {
      *(outputIter++) = static_cast<stream_type>(state);
      state >>= this->getStreamOutTypeBits();
      assert(state < maxState);
    }

    return {state, outputIter};
  };

  SingleStreamEncoderImplBase() = default;
  explicit SingleStreamEncoderImplBase(size_t symbolTablePrecision) noexcept : base_type{symbolTablePrecision} {};
};

template <size_t lowerBound_V>
class CompatEncoderImpl : public SingleStreamEncoderImplBase<lowerBound_V, const Symbol*, CompatEncoderImpl<lowerBound_V>>
{
  using base_type = SingleStreamEncoderImplBase<lowerBound_V, const Symbol*, CompatEncoderImpl<lowerBound_V>>;

 public:
  using stream_type = typename base_type::stream_type;
  using state_type = typename base_type::state_type;
  using symbol_type = typename base_type::symbol_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;

 public:
  CompatEncoderImpl() = default;
  explicit CompatEncoderImpl(size_type symbolTablePrecission) noexcept : base_type{symbolTablePrecission} {};

  using base_type::putSymbols;

  template <typename stream_IT>
  [[nodiscard]] inline stream_IT putSymbols(stream_IT outputIter, const symbol_type& symbol)
  {
    assert(symbol->getFrequency() != 0);

    const auto [newState, streamPosition] = this->renorm(this->mState, outputIter, symbol->getFrequency());
    // coding function
    this->mState = ((newState / symbol->getFrequency()) << this->mSymbolTablePrecision) + symbol->getCumulative() + (newState % symbol->getFrequency());

    return streamPosition;
  };
};

#ifdef RANS_SINGLE_STREAM
template <size_t lowerBound_V>
class SingleStreamEncoderImpl : public SingleStreamEncoderImplBase<lowerBound_V, const PrecomputedSymbol*, SingleStreamEncoderImpl<lowerBound_V>>
{
  using base_type = SingleStreamEncoderImplBase<lowerBound_V, const PrecomputedSymbol*, SingleStreamEncoderImpl<lowerBound_V>>;
  __extension__ using uint128_t = unsigned __int128;

 public:
  using stream_type = typename base_type::stream_type;
  using state_type = typename base_type::state_type;
  using symbol_type = typename base_type::symbol_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;

 public:
  SingleStreamEncoderImpl() = default;
  explicit SingleStreamEncoderImpl(size_type symbolTablePrecission) noexcept : base_type{symbolTablePrecission} {};

  using base_type::putSymbols;

  template <typename stream_IT>
  [[nodiscard]] inline stream_IT putSymbols(stream_IT outputIter, const symbol_type& symbol)
  {
    assert(symbol->getFrequency() != 0);

    const auto [newState, streamPosition] = this->renorm(this->mState, outputIter, symbol->getFrequency());
    // coding function
    state_type quotient = static_cast<state_type>((static_cast<uint128_t>(newState) * symbol->getReciprocalFrequency()) >> 64);
    quotient = quotient >> symbol->getReciprocalShift();
    this->mState = newState + symbol->getCumulative() + quotient * symbol->getFrequencyComplement();

    return streamPosition;
  };
};
#endif /* RANS_SINGLE_STREAM */

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_ENCODE_SINGLESTREAMENCODERIMPL_H_ */