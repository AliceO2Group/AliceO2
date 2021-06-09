// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "rANS/internal/Encoder.h"
#include "rANS/internal/EncoderSymbol.h"
#include "rANS/internal/helper.h"
#include "rANS/internal/SymbolTable.h"
#include "rANS/FrequencyTable.h"

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
  using encoderSymbolTable_t = typename internal::SymbolTable<internal::EncoderSymbol<coder_T>>;

 public:
  using symbol_t = typename FrequencyTable::symbol_t;

  using coder_t = coder_T;
  using stream_t = stream_T;
  using source_t = source_T;

  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  EncoderBase() noexcept {}; //NOLINT
  EncoderBase(encoderSymbolTable_t&& e, size_t symbolTablePrecission) noexcept;
  EncoderBase(const FrequencyTable& frequencies, size_t symbolTablePrecission);

  inline size_t getSymbolTablePrecision() const noexcept { return mSymbolTablePrecission; }
  inline size_t getAlphabetRangeBits() const noexcept { return mSymbolTable.getAlphabetRangeBits(); }
  inline symbol_t getMinSymbol() const noexcept { return mSymbolTable.getMinSymbol(); }
  inline symbol_t getMaxSymbol() const noexcept { return mSymbolTable.getMaxSymbol(); }

 protected:
  encoderSymbolTable_t mSymbolTable{};
  size_t mSymbolTablePrecission{};

  using ransCoder_t = typename internal::Encoder<coder_T, stream_T>;
};

template <typename coder_T, typename stream_T, typename source_T>
EncoderBase<coder_T, stream_T, source_T>::EncoderBase(encoderSymbolTable_t&& e, size_t symbolTablePrecission) noexcept : mSymbolTable{std::move(e)}, mSymbolTablePrecission{symbolTablePrecission} {};

template <typename coder_T, typename stream_T, typename source_T>
EncoderBase<coder_T, stream_T, source_T>::EncoderBase(const FrequencyTable& frequencies,
                                                      size_t symbolTablePrecission)
{
  SymbolStatistics stats{frequencies, mSymbolTablePrecission};
  mSymbolTablePrecission = stats.getSymbolTablePrecision();

  RANSTimer t;
  t.start();
  mSymbolTable = encoderSymbolTable_t{stats};
  t.stop();
  LOG(debug1) << "Encoder SymbolTable inclusive time (ms): " << t.getDurationMS();
}

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_INTERNAL_ENCODERBASE_H_ */
