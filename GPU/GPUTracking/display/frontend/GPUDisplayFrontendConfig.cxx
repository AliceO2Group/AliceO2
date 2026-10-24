// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayFrontendConfig.cxx
/// \author David Rohr

#include "GPUDisplayFrontend.h"
#include "GPUDisplay.h"
#include "frontend/GPUDisplayInfo.inc"
#include "GPULogging.h"
#include "utils/qconfig_rtti.h"
#include "utils/qconfig_internal_helpers.h"

#include <algorithm>
#include <concepts>
#include <cstring>
#include <limits>
#include <optional>
#include <array>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

using namespace o2::gpu;
struct GPUDisplayFrontend::configRTTIExt : public qConfig::qConfigRTTIentry {
};

using configKeyCallback = std::function<void()>;

struct GPUDisplayFrontend::configKey {
  uint32_t configRTTIIndex;
  configKeyCallback callback;
};

#ifdef GPUCA_DISPLAY_OPENGL_CORE
static constexpr bool displayOpenGLCore = true; // OpenGL compat profile not available
#else
static constexpr bool displayOpenGLCore = false;
#endif

struct GPUDisplayFrontend::configRTTIInfo {
  static constexpr std::tuple configStructs{&GPUDisplay::mCfgL, &GPUDisplay::mCfgH, &GPUDisplay::mCfgR}; // Config structs with RTTI, as members of GPUDisplay
  std::vector<configRTTIExt> rttiEntries;
  std::vector<std::vector<uint32_t>> rttiReverseId; // [configStructIndex][offset] -> index in rttiEntries, INVALID_INDEX if no member at that offset
  static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();
  std::vector<std::string> configStructNames;
  std::array<configKey, N_TOTAL_KEYS> configKeys; // By key code, special keys with SHIFT at code + SPECIAL_KEYS_SHIFTED
  std::vector<std::string> keyNames;
  std::vector<std::pair<std::string, std::string>> helpTextEntries; // Keys and text of each add_key_binding call
  std::string helpText;
  static constexpr std::string_view possibleKeys = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#.:;>'^_"; // Bindable keys >= 32 (space), special keys < 32 are those with a name in keyNames
  static constexpr char reservedKeysData[] = {'W', 'A', 'S', 'D', 'E', 'F', 'w', 'a', 's', 'd', 'e', 'f', '+', '-', '<', KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_PAGEUP, KEY_PAGEDOWN, KEY_SHIFT, KEY_ALT, KEY_RALT, KEY_CTRL, KEY_RCTRL};
  static constexpr std::string_view reservedKeys{reservedKeysData, sizeof(reservedKeysData)}; // Keys queried continuously as held keys (camera control, modifiers), not available for bindings

  template <class T, size_t I = 0>
  static constexpr uint32_t configStructIndex() // Index of struct T in configStructs, compile-time error if not present
  {
    if constexpr (std::is_same_v<std::tuple_element_t<I, std::remove_cv_t<decltype(configStructs)>>, T GPUDisplay::*>) {
      return I;
    } else {
      return configStructIndex<T, I + 1>();
    }
  }
  template <class T>
  static T& configStruct(GPUDisplay& display)
  {
    return display.*std::get<configStructIndex<T>()>(configStructs);
  }
  template <class T, class M>
  uint32_t getRTTIIndex(GPUDisplay& display, M T::* member) // RTTI index of a direct member of a config struct
  {
    if constexpr (std::is_same_v<T, GPUDisplay::control>) {
      return INVALID_INDEX;
    } else {
      const T& obj = configStruct<T>(display);
      return rttiReverseId[configStructIndex<T>()][(const char*)&(obj.*member) - (const char*)&obj];
    }
  }
  void buildHelpText() // Keys left-aligned, the ':' aligned after the longest keys
  {
    size_t width = 0;
    for (const auto& e : helpTextEntries) {
      width = std::max(width, e.first.size());
    }
    helpText.clear();
    for (const auto& e : helpTextEntries) {
      helpText += e.first + std::string(width - e.first.size(), ' ') + ": " + e.second + "\n";
    }
  }
  std::string getKeyName(uint16_t key) const // Name of a special key (KEY_*), the character otherwise, the code if neither, SHIFT+ for shifted special keys
  {
    if (key >= 256) {
      return "SHIFT+" + getKeyName(key - SPECIAL_KEYS_SHIFTED);
    }
    return key < keyNames.size() && !keyNames[key].empty() ? keyNames[key] : (key < 32 ? std::to_string(key) : std::string(1, (char)key));
  }
};

namespace
{
#include "utils/qconfig_genrtti.inc"
} // namespace

void GPUDisplay::checkAndApplyConfigChanges(const GPUSettingsDisplayHeavy& oldCfgH, const GPUSettingsDisplayLight& oldCfgL, const GPUSettingsDisplayRenderer& oldCfgR)
{
  if (memcmp((void*)&oldCfgH, (void*)&mCfgH, sizeof(mCfgH)) != 0) {
    mUpdateEventData = true;
  }
  if (mCfgL.propagateTracks != 0 && !mTracksArePropagated) {
    mUpdateVertexLists = true;
  }
  if (memcmp((void*)&oldCfgL, (void*)&mCfgL, sizeof(mCfgL)) != 0 || memcmp((void*)&oldCfgR, (void*)&mCfgR, sizeof(mCfgR)) != 0) {
    mUpdateDrawCommands = true;
  }
  if (oldCfgR.drawQualityMSAA != mCfgR.drawQualityMSAA || oldCfgR.drawQualityDownsampleFSAA != mCfgR.drawQualityDownsampleFSAA || oldCfgL.depthBuffer != mCfgL.depthBuffer || oldCfgR.screenshotScaleFactor != mCfgR.screenshotScaleFactor) {
    mUpdateRenderPipeline = true;
  }

  if (oldCfgR.drawQualityVSync != mCfgR.drawQualityVSync) {
    mFrontend->SetVSync(mCfgR.drawQualityVSync);
    mBackend->SetVSync(mCfgR.drawQualityVSync);
  }
  if (oldCfgR.fullScreen != mCfgR.fullScreen) {
    mFrontend->SwitchFullscreen(mCfgR.fullScreen);
  }
  if (oldCfgR.maximized != mCfgR.maximized) {
    mFrontend->ToggleMaximized(mCfgR.maximized);
  }
  if (oldCfgR.useGLIndirectDraw != mCfgR.useGLIndirectDraw) {
    mUpdateEventData = true;
  }
  if (oldCfgL.animationMode != mCfgL.animationMode) {
    resetAnimation();
  }
  if (oldCfgH.trackFilter != mCfgH.trackFilter) {
    mUpdateTrackFilter = true;
  }
}

void GPUDisplay::HandleExternalGUIKey(int32_t key)
{
  // GPUError("key %d '%c'", key, (char) key);

  bool shifted = key >= 'A' && key <= 'Z';
  bool oldShift = mFrontend->mKeys[mFrontend->KEY_SHIFT];
  mFrontend->mKeys[mFrontend->KEY_SHIFT] = shifted;
  HandleKey(key, true);
  mFrontend->mKeys[mFrontend->KEY_SHIFT] = oldShift;
}

bool GPUDisplayFrontend::HasExternalGUISignals() const
{
  return mExternalGUIKeys.size();
}

bool GPUDisplayFrontend::HandleExternalGUIKeys()
{
  std::lock_guard lock(mExternalGUIKeyMutex);
  for (auto key : mExternalGUIKeys) {
    mDisplay->HandleExternalGUIKey(key);
  }
  return !std::exchange(mExternalGUIKeys, {}).empty();
}

template <typename... Args>
void GPUDisplayFrontend::add_key_binding(Args&&... args)
{
  constexpr size_t nKeys = (0 + ... + std::is_same_v<std::remove_cvref_t<Args>, std::tuple<uint16_t, uint32_t, configKeyCallback>>);
  static_assert(nKeys >= 1 && (sizeof...(Args) == nKeys + 1 || sizeof...(Args) == nKeys + 2), "add_key_binding needs at least one key, the help text, and optionally the keys to show in the help text");
  auto argTuple = std::forward_as_tuple(args...);
  std::string keys;
  auto addKey = [&](std::tuple<uint16_t, uint32_t, configKeyCallback>& key) {
    const std::string name = mConfigRTTI->getKeyName(std::get<0>(key));
    auto& entry = mConfigRTTI->configKeys[std::get<0>(key)];
    if (entry.callback) {
      throw std::runtime_error("Key [" + name + "] assigned twice in display frontend");
    }
    entry.configRTTIIndex = std::get<1>(key);
    entry.callback = std::move(std::get<2>(key));
    keys += (keys.empty() ? "[" : " [") + name + "]";
  };
  [&]<size_t... I>(std::index_sequence<I...>) { (addKey(std::get<I>(argTuple)), ...); }(std::make_index_sequence<nKeys>());
  if constexpr (sizeof...(Args) == nKeys + 2) {
    keys = std::get<nKeys + 1>(argTuple);
  }
  mConfigRTTI->helpTextEntries.emplace_back(std::move(keys), std::get<nKeys>(argTuple));
}

void GPUDisplayFrontend::listConfig()
{
  const auto& rtti = *mConfigRTTI;
  auto keyOrder = [](uint16_t k) { return std::pair(k >= '0' && k <= '9' ? 0 : k >= 'a' && k <= 'z' ? 1
                                                                             : k >= 'A' && k <= 'Z' ? 2
                                                                             : k >= 256             ? 5
                                                                             : k > 32               ? 3
                                                                                                    : 4,
                                                    k); }; // 0-9, a-z, A-Z, other characters, special keys, special keys with SHIFT
  std::vector<uint16_t> boundKeys, availableKeys;
  for (uint32_t k = 1; k < rtti.configKeys.size(); k++) {
    const uint32_t base = k >= 256 ? k - SPECIAL_KEYS_SHIFTED : k; // A shifted special key is possible if the special key is
    const bool possible = (base < N_SPECIAL_KEYS && !rtti.keyNames[base].empty()) || (k < 256 && rtti.possibleKeys.find((char)k) != std::string_view::npos);
    if (rtti.configKeys[k].callback) {
      boundKeys.emplace_back(k);
    } else if (possible && rtti.reservedKeys.find((char)base) == std::string_view::npos) {
      availableKeys.emplace_back(k);
    }
  }
  std::sort(boundKeys.begin(), boundKeys.end(), [&](uint16_t a, uint16_t b) { return keyOrder(a) < keyOrder(b); });
  std::sort(availableKeys.begin(), availableKeys.end(), [&](uint16_t a, uint16_t b) { return keyOrder(a) < keyOrder(b); });

  GPUInfo("Display config options with RTTI (struct, option, type, keys):");
  uint32_t nOptionsWithKey = 0;
  for (uint32_t i = 0; i < rtti.rttiEntries.size(); i++) {
    const auto& e = rtti.rttiEntries[i];
    std::string keys;
    for (uint16_t k : boundKeys) {
      if (rtti.configKeys[k].configRTTIIndex == i) {
        keys += (keys.empty() ? "[" : " [") + rtti.getKeyName(k) + "]";
      }
    }
    nOptionsWithKey += !keys.empty();
    GPUInfo("  %-28s %-32s %-12s %s", e.configStruct.c_str(), e.name.c_str(), e.type.c_str(), keys.c_str());
  }
  GPUInfo("Bound keys (key, struct, option):");
  for (uint16_t k : boundKeys) {
    const uint32_t index = rtti.configKeys[k].configRTTIIndex;
    const bool hasRTTI = index != configRTTIInfo::INVALID_INDEX;
    GPUInfo("  %-10s %-28s %s", ("[" + rtti.getKeyName(k) + "]").c_str(), hasRTTI ? rtti.rttiEntries[index].configStruct.c_str() : "(no config option)", hasRTTI ? rtti.rttiEntries[index].name.c_str() : "");
  }
  std::string available;
  for (uint16_t k : availableKeys) {
    available += (available.empty() ? "[" : " [") + rtti.getKeyName(k) + "]";
  }
  GPUInfo("Available keys: %s", available.c_str());
  GPUInfo("%zu options with RTTI, %u of them with a key, %zu bound keys, %zu available keys", rtti.rttiEntries.size(), nOptionsWithKey, boundKeys.size(), availableKeys.size());
}

#include "GPUDisplayFrontendConfigKeys.inc"

void GPUDisplay::buildConfig()
{
  auto& rtti = mFrontend->mConfigRTTI;
  rtti = {new GPUDisplayFrontend::configRTTIInfo, [](GPUDisplayFrontend::configRTTIInfo* p) { delete p; }};

  rtti->keyNames.resize(rtti->configKeys.size());
#define GPUDisplayFrontendAddKey(code, key) rtti->keyNames[code] = #key
#include "frontend/GPUDisplayFrontendKeys.h"
#undef GPUDisplayFrontendAddKey

  std::tie(rtti->rttiEntries, rtti->configStructNames) = std::apply([this](auto... s) { return qConfigGenRTTI<GPUDisplayFrontend::configRTTIExt, std::remove_reference_t<decltype(this->*s)>...>(); }, rtti->configStructs);
  std::apply([&](auto... s) { (rtti->rttiReverseId.emplace_back(sizeof(this->*s), GPUDisplayFrontend::configRTTIInfo::INVALID_INDEX), ...); }, rtti->configStructs);
  for (uint32_t i = 0; i < rtti->rttiEntries.size(); i++) {
    const auto& entry = rtti->rttiEntries[i];
    rtti->rttiReverseId[entry.configStructIndex][entry.offset] = i;
  }

  buildConfigKeys();
  rtti->buildHelpText();
}

bool GPUDisplayFrontend::processKeyLambda(uint32_t key)
{
  if (key < N_SPECIAL_KEYS && mKeys[KEY_SHIFT]) {
    key += SPECIAL_KEYS_SHIFTED;
  }
  if (key < mConfigRTTI->configKeys.size() && mConfigRTTI->configKeys[key].callback) {
    mConfigRTTI->configKeys[key].callback();
    return true;
  }
  return false;
}
