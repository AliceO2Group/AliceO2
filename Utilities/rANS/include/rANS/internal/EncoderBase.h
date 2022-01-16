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

/// @file   EncoderBase.h
/// @author michael.lettrich@cern.ch
/// @since  Feb 8, 2021
/// @brief

#ifndef INCLUDE_RANS_INTERNAL_ENCODERBASE_H_
#define INCLUDE_RANS_INTERNAL_ENCODERBASE_H_

#include <memory>
#include <algorithm>
#include <iomanip>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include "rANS/definitions.h"
#include "rANS/internal/helper.h"
#include "rANS/internal/backend/cpp/Encoder.h"
#include "rANS/internal/backend/cpp/EncoderSymbol.h"
#include "rANS/internal/SymbolTable.h"
#include "rANS/RenormedFrequencyTable.h"

namespace o2
{
namespace rans
{
namespace internal
{

template <typename coder_T, typename stream_T, typename source_T>
class EncoderBase
{
 protected:
  using encoderSymbolTable_t = typename internal::SymbolTable<cpp::EncoderSymbol<coder_T>>;
  using ransCoder_t = typename cpp::Encoder<coder_T, stream_T>;

 public:
  using coder_t = coder_T;
  using stream_t = stream_T;
  using source_t = source_T;

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  EncoderBase() noexcept {}; // NOLINT
  explicit EncoderBase(const RenormedFrequencyTable& frequencyTable) : mSymbolTable{frequencyTable} {};

  inline size_t getSymbolTablePrecision() const noexcept { return mSymbolTable.getPrecision(); };
  inline size_t getAlphabetRangeBits() const noexcept { return mSymbolTable.getAlphabetRangeBits(); };
  inline symbol_t getMinSymbol() const noexcept { return mSymbolTable.getMinSymbol(); };
  inline symbol_t getMaxSymbol() const noexcept { return mSymbolTable.getMaxSymbol(); };

 protected:
  encoderSymbolTable_t mSymbolTable{};
};

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_INTERNAL_ENCODERBASE_H_ */
