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

// \file FeeConfiguration.h
/// \brief Utilities to describe FEE configuration
/// \author wiktor.pierozak@cern.ch

#ifndef O2_FIT_FEE_CONFIGURATION
#define O2_FIT_FEE_CONFIGURATION

#include <gsl/span>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <Rtypes.h>

namespace o2::fit
{
namespace config_helpers
{
struct DefaultValueType {
  template <typename T>
    requires std::is_arithmetic_v<T>
  constexpr operator T() const noexcept
  {
    if constexpr (std::is_same_v<T, bool>) {
      return false;
    } else {
      return std::numeric_limits<T>::max();
    }
  }
};

inline constexpr DefaultValueType DefaultValue{};

template <typename T, std::size_t N>
constexpr void fillDefaultArray(T (&array)[N])
{
  for (auto& element : array) {
    element = DefaultValue;
  }
}
} // namespace config_helpers

struct TcmConfig {
  float phaseDelayA{config_helpers::DefaultValue};
  float phaseDelayC{config_helpers::DefaultValue};

  bool operator==(const TcmConfig&) const = default;
  ClassDefNV(TcmConfig, 1);
};

struct PmConfig {
  uint8_t orGate{config_helpers::DefaultValue};
  uint16_t trgChargeHighLevel{config_helpers::DefaultValue};
  uint16_t trgChargeLowLevel{config_helpers::DefaultValue};
  bool fddCoincidenceMode{config_helpers::DefaultValue};
  bool operator==(const PmConfig&) const = default;
  ClassDefNV(PmConfig, 1);
};

template <int NChannels>
struct ChannelsConfig {
  int16_t timeAligments[NChannels];
  uint16_t cfdThresholds[NChannels];
  int16_t cfdZeros[NChannels];
  int16_t adcZeros[NChannels];
  uint16_t adcDelays[NChannels];
  uint16_t rangeCorrectionAdc0[NChannels];
  uint16_t rangeCorrectionAdc1[NChannels];
  uint16_t thresholdCalibration[NChannels];
  bool channelMaskData[NChannels];
  bool channelMaskTriggers[NChannels];

  constexpr ChannelsConfig()
  {
    config_helpers::fillDefaultArray(timeAligments);
    config_helpers::fillDefaultArray(cfdThresholds);
    config_helpers::fillDefaultArray(cfdZeros);
    config_helpers::fillDefaultArray(adcZeros);
    config_helpers::fillDefaultArray(adcDelays);
    config_helpers::fillDefaultArray(rangeCorrectionAdc0);
    config_helpers::fillDefaultArray(rangeCorrectionAdc1);
    config_helpers::fillDefaultArray(channelMaskData);
    config_helpers::fillDefaultArray(channelMaskTriggers);
    config_helpers::fillDefaultArray(thresholdCalibration);
  }

  template <typename T>
  [[nodiscard]] static constexpr bool isDefault(const T& value)
  {
    return value == static_cast<T>(config_helpers::DefaultValue);
  }

  gsl::span<const int16_t, NChannels> getTimeAligments() const { return timeAligments; }
  gsl::span<const uint16_t, NChannels> getCfdThresholds() const { return cfdThresholds; }
  gsl::span<const int16_t, NChannels> getCfdZeros() const { return cfdZeros; }
  gsl::span<const int16_t, NChannels> getAdcZeros() const { return adcZeros; }
  gsl::span<const uint16_t, NChannels> getAdcDelays() const { return adcDelays; }
  gsl::span<const uint16_t, NChannels> getRangeCorrectionAdc0() const { return rangeCorrectionAdc0; }
  gsl::span<const uint16_t, NChannels> getRangeCorrectionAdc1() const { return rangeCorrectionAdc1; }
  gsl::span<const bool, NChannels> getChannelMaskTriggers() const { return channelMaskTriggers; }
  gsl::span<const bool, NChannels> getChannelMaskData() const { return channelMaskData; }

  bool operator==(const ChannelsConfig& other) const
  {
    return std::equal(std::begin(timeAligments), std::end(timeAligments), std::begin(other.timeAligments)) &&
           std::equal(std::begin(cfdThresholds), std::end(cfdThresholds), std::begin(other.cfdThresholds)) &&
           std::equal(std::begin(cfdZeros), std::end(cfdZeros), std::begin(other.cfdZeros)) &&
           std::equal(std::begin(adcZeros), std::end(adcZeros), std::begin(other.adcZeros)) &&
           std::equal(std::begin(adcDelays), std::end(adcDelays), std::begin(other.adcDelays)) &&
           std::equal(std::begin(rangeCorrectionAdc0), std::end(rangeCorrectionAdc0), std::begin(other.rangeCorrectionAdc0)) &&
           std::equal(std::begin(rangeCorrectionAdc1), std::end(rangeCorrectionAdc1), std::begin(other.rangeCorrectionAdc1)) &&
           std::equal(std::begin(channelMaskData), std::end(channelMaskData), std::begin(other.channelMaskData)) &&
           std::equal(std::begin(channelMaskTriggers), std::end(channelMaskTriggers), std::begin(other.channelMaskTriggers));
  }

  ClassDefNV(ChannelsConfig<NChannels>, 1);
};

template <std::size_t NChannels>
struct HvChannelsConfig {
  float gain[NChannels];

  constexpr HvChannelsConfig()
  {
    config_helpers::fillDefaultArray(gain);
  }
  bool operator==(const HvChannelsConfig& other) const
  {
    return std::equal(std::begin(gain), std::end(gain), std::begin(other.gain));
  }
  ClassDefNV(HvChannelsConfig<NChannels>, 1);
};
} // namespace o2::fit
#endif