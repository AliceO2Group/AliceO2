// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawReaderFIT.h class  for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FIT data blocks and push them to DigitBlockFIT for proccess
#ifndef ALICEO2_FIT_RAWREADERFIT_H_
#define ALICEO2_FIT_RAWREADERFIT_H_
#include <iostream>
#include <vector>
#include <type_traits>
#include <Rtypes.h>
#include "FITRaw/RawReaderBaseFIT.h"
#include "FITRaw/DigitBlockFIT.h"
#include <Framework/Logger.h>
#include "Framework/ProcessingContext.h"
#include "Framework/DataAllocator.h"
#include "Framework/OutputSpec.h"
#include <gsl/span>

namespace o2
{
namespace fit
{

template <typename RawReaderType, bool useTrgInput = false>
class RawReaderFIT : public RawReaderType
{
 public:
  RawReaderFIT(o2::header::DataOrigin dataOrigin, bool dumpData) : mDataOrigin(dataOrigin), mDumpData(dumpData) {}
  RawReaderFIT(const RawReaderFIT&) = default;
  RawReaderFIT() = delete;
  ~RawReaderFIT() = default;
  typedef RawReaderType RawReader_t;
  typedef typename RawReader_t::DigitBlockFIT_t DigitBlockFIT_t;
  typedef typename DigitBlockFIT_t::LookupTable_t LookupTable_t;
  typedef typename DigitBlockFIT_t::Digit_t Digit_t;
  typedef typename DigitBlockFIT_t::SubDigit_t SubDigitTmp_t;
  typedef typename DigitBlockHelper::GetSubDigitField<typename DigitBlockFIT_t::VecSingleSubDigit_t>::vector_type SingleSubDigitTmp_t;
  typedef typename Digit_t::DetTrigInput_t DetTrigInput_t;
  typedef std::make_index_sequence<DigitBlockFIT_t::sNSubDigits> IndexesSubDigit;
  typedef std::make_index_sequence<DigitBlockFIT_t::sNSingleSubDigits> IndexesSingleSubDigit;

  static constexpr bool sSubDigitExists = !std::is_same<SubDigitTmp_t, std::tuple<>>::value;
  static constexpr bool sSingleSubDigitExists = !std::is_same<SingleSubDigitTmp_t, std::tuple<>>::value;
  //Wrapping by std::tuple
  typedef typename std::conditional<DigitBlockFIT_t::sNSubDigits != 1, SubDigitTmp_t, std::tuple<SubDigitTmp_t>>::type SubDigit_t;
  typedef typename std::conditional<DigitBlockFIT_t::sNSingleSubDigits != 1, SingleSubDigitTmp_t, std::tuple<SingleSubDigitTmp_t>>::type SingleSubDigit_t;

  static constexpr bool sUseTrgInput = useTrgInput;
  o2::header::DataOrigin mDataOrigin;
  std::vector<Digit_t> mVecDigit;
  std::vector<DetTrigInput_t> mVecTrgInput;
  SubDigit_t mVecSubDigit;             //tuple of vectors
  SingleSubDigit_t mVecSingleSubDigit; //tuple of vectors
  bool mDumpData;

  void clear()
  {
    mVecDigit.clear();
    if constexpr (sUseTrgInput) {
      mVecTrgInput.clear();
    }
    if constexpr (sSubDigitExists) {
      std::apply([](auto&... subDigit) {
        ((subDigit.clear()), ...);
      },
                 mVecSubDigit);
    }
    if constexpr (sSingleSubDigitExists) {
      std::apply([](auto&... singleSubDigit) {
        ((singleSubDigit.clear()), ...);
      },
                 mVecSingleSubDigit);
    }
  }
  template <std::size_t... IsubDigits, std::size_t... IsingleSubDigits>
  auto callGetDigit(std::index_sequence<IsubDigits...>, std::index_sequence<IsingleSubDigits...>)
  {
    if constexpr (sUseTrgInput) {
      RawReader_t::getDigits(mVecDigit, std::get<IsubDigits>(mVecSubDigit)..., std::get<IsingleSubDigits>(mVecSingleSubDigit)..., mVecTrgInput);
    } else {
      RawReader_t::getDigits(mVecDigit, std::get<IsubDigits>(mVecSubDigit)..., std::get<IsingleSubDigits>(mVecSingleSubDigit)...);
    }
  }

  template <std::size_t... IsubDigits, std::size_t... IsingleSubDigits>
  auto callPrint(std::index_sequence<IsubDigits...>, std::index_sequence<IsingleSubDigits...>) const
  {
    DigitBlockFIT_t::print(mVecDigit, std::get<IsubDigits>(mVecSubDigit)..., std::get<IsingleSubDigits>(mVecSingleSubDigit)...);
  }
  void accumulateDigits()
  {
    callGetDigit(IndexesSubDigit{}, IndexesSingleSubDigit{});
    LOG(INFO) << "Number of Digits: " << mVecDigit.size();
    /*
    if constexpr (sSubDigitExists) {
      std::apply([](const auto&... subDigit) {
            ((LOG(INFO)<<"Total "<<std::decay<decltype(subDigit)>::type::value_type::sDigitName<<":"<<subDigit.size()), ...);
        },
      mVecSubDigit);
    }
    if constexpr (sSingleSubDigitExists) {
      std::apply([](const auto&... singleSubDigit) {
            ((LOG(INFO)<<"Total "<<std::decay<decltype(singleSubDigit)>::type::value_type::sDigitName<<":"<<singleSubDigit.size()), ...);
        },
      mVecSingleSubDigit);
    }
    */
    if (mDumpData) {
      callPrint(IndexesSubDigit{}, IndexesSingleSubDigit{});
    }
  }
  void configureOutputSpec(std::vector<o2::framework::OutputSpec>& outputSpec) const
  {
    outputSpec.emplace_back(mDataOrigin, Digit_t::sChannelNameDPL, 0, o2::framework::Lifetime::Timeframe);
    if constexpr (sSubDigitExists) {
      std::apply([&](const auto&... subDigit) {
        ((outputSpec.emplace_back(mDataOrigin, (std::decay<decltype(subDigit)>::type::value_type::sChannelNameDPL), 0, o2::framework::Lifetime::Timeframe)), ...);
      },
                 mVecSubDigit);
    }
    if constexpr (sSingleSubDigitExists) {
      std::apply([&](const auto&... singleSubDigit) {
        ((outputSpec.emplace_back(mDataOrigin, (std::decay<decltype(singleSubDigit)>::type::value_type::sChannelNameDPL), 0, o2::framework::Lifetime::Timeframe)), ...);
      },
                 mVecSingleSubDigit);
    }
    if constexpr (sUseTrgInput) {
      outputSpec.emplace_back(mDataOrigin, DetTrigInput_t::sChannelNameDPL, 0, o2::framework::Lifetime::Timeframe);
    }
  }
  void makeSnapshot(o2::framework::ProcessingContext& pc) const
  {
    pc.outputs().snapshot(o2::framework::Output{mDataOrigin, Digit_t::sChannelNameDPL, 0, o2::framework::Lifetime::Timeframe}, mVecDigit);
    if constexpr (sSubDigitExists) {
      std::apply([&](const auto&... subDigit) {
        ((pc.outputs().snapshot(o2::framework::Output{mDataOrigin, (std::decay<decltype(subDigit)>::type::value_type::sChannelNameDPL), 0, o2::framework::Lifetime::Timeframe}, subDigit)), ...);
      },
                 mVecSubDigit);
    }
    if constexpr (sSingleSubDigitExists) {
      std::apply([&](const auto&... singleSubDigit) {
        ((pc.outputs().snapshot(o2::framework::Output{mDataOrigin, (std::decay<decltype(singleSubDigit)>::type::value_type::sChannelNameDPL), 0, o2::framework::Lifetime::Timeframe}, singleSubDigit)), ...);
      },
                 mVecSingleSubDigit);
    }
    if constexpr (sUseTrgInput) {
      pc.outputs().snapshot(o2::framework::Output{mDataOrigin, DetTrigInput_t::sChannelNameDPL, 0, o2::framework::Lifetime::Timeframe}, mVecTrgInput);
    }
  }
};

} // namespace fit
} // namespace o2

#endif