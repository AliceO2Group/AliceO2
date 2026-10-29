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
  uint32_t configRTTIIndex = std::numeric_limits<uint32_t>::max(); // Index in rttiEntries, KEY_MAP_INDEX + id for a key down mapping, INVALID_INDEX otherwise
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
  std::vector<configKeyCallback> configInfos;       // SetInfo callback for an option, indexed like rttiEntries, called after the callback of a key bound to it
  std::vector<std::vector<uint32_t>> rttiReverseId; // [configStructIndex][offset] -> index in rttiEntries, INVALID_INDEX if no member at that offset
  static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t KEY_MAP_INDEX = 0x80000000; // configRTTIIndex of keys mapped to mKeyDownMap[configRTTIIndex - KEY_MAP_INDEX]
  static bool isKeyMap(uint32_t index) { return index >= KEY_MAP_INDEX && index != INVALID_INDEX; }
  std::vector<std::string> configStructNames;
  std::array<configKey, N_TOTAL_KEYS> configKeys; // By key code, special keys with SHIFT at code + SPECIAL_KEYS_SHIFTED
  std::vector<std::string> keyNames;
  std::vector<std::string> keyMapNames;                             // Names of the key down mapping ids (KEY_MAP_*)
  std::vector<std::pair<std::string, std::string>> helpTextEntries; // Keys and text of each add_key_binding call
  std::vector<std::string> helpText;
  static constexpr std::string_view possibleKeys = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#.:;>'^_"; // Bindable keys >= 32 (space), special keys < 32 are those with a name in keyNames

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
  static char* configStructPtr(GPUDisplay& display, uint32_t index) // Runtime version of configStruct by index
  {
    char* retVal = nullptr;
    uint32_t i = 0;
    std::apply([&](auto... s) { ((retVal = i++ == index ? reinterpret_cast<char*>(&(display.*s)) : retVal), ...); }, configStructs);
    return retVal;
  }
  std::optional<std::string> valueString(GPUDisplay& display, uint32_t index) const // Current value of an option, printed with print_type as in qconfig.cxx, nullopt for unsupported types
  {
    const auto& e = rttiEntries[index];
    const char* ptr = configStructPtr(display, e.configStructIndex) + e.offset;
    if (e.isVector || e.arraySize) {
      return std::nullopt;
    } else if (e.type == "bool") {
      return qConfig::print_type(*(const bool*)ptr);
    } else if (e.type == "float") {
      return qConfig::print_type(*(const float*)ptr);
    } else if (e.type == "int8_t") {
      return qConfig::print_type(*(const int8_t*)ptr);
    } else if (e.type == "uint8_t") {
      return qConfig::print_type(*(const uint8_t*)ptr);
    } else if (e.type == "int32_t" || e.type == "int") {
      return qConfig::print_type(*(const int32_t*)ptr);
    } else if (e.type == "uint32_t") {
      return qConfig::print_type(*(const uint32_t*)ptr);
    }
    return std::nullopt;
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
      helpText.emplace_back(e.first + std::string(width - e.first.size(), ' ') + ": " + e.second);
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

void GPUDisplay::checkAndApplyConfigChanges()
{
  const auto& [oldCfgH, oldCfgL, oldCfgR] = mOldCfg;
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
    mFrontend->requestFullScreen(mCfgR.fullScreen);
  }
  if (oldCfgR.maximized != mCfgR.maximized) {
    mFrontend->requestMaximized(mCfgR.maximized);
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
  HandleKey(key);
  mFrontend->mKeys[mFrontend->KEY_SHIFT] = oldShift;
}

bool GPUDisplayFrontend::HasExternalGUISignals() const
{
  return mExternalGUIKeys.size();
}

void GPUDisplayFrontend::HandleExternalGUIKeys()
{
  std::lock_guard lock(mExternalGUIKeyMutex);
  for (auto key : mExternalGUIKeys) {
    mDisplay->HandleExternalGUIKey(key);
  }
  mExternalGUIKeys.clear();
}

template <typename... Args>
void GPUDisplayFrontend::add_key_binding(Args&&... args)
{
  constexpr size_t nKeys = (0 + ... + (std::is_same_v<std::remove_cvref_t<Args>, std::tuple<uint16_t, uint32_t, configKeyCallback>> || std::is_same_v<std::remove_cvref_t<Args>, std::tuple<uint8_t, uint8_t>>));
  static_assert(nKeys >= 1 && (sizeof...(Args) == nKeys + 1 || sizeof...(Args) == nKeys + 2), "add_key_binding needs at least one key, the help text, and optionally the keys to show in the help text");
  auto argTuple = std::forward_as_tuple(args...);
  std::string keys;
  auto addKey = [&](auto& key) {
    const uint16_t character = std::get<0>(key);
    const std::string name = mConfigRTTI->getKeyName(character);
    auto& entry = mConfigRTTI->configKeys[character];
    if (entry.callback || entry.configRTTIIndex != configRTTIInfo::INVALID_INDEX) {
      throw std::runtime_error("Key [" + name + "] assigned twice in display frontend");
    }
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(key)>, std::tuple<uint8_t, uint8_t>>) {
      const uint8_t id = std::get<1>(key);
      if (mKeyDownMap[id]) {
        throw std::runtime_error("Key down mapping " + mConfigRTTI->keyMapNames[id] + " assigned twice in display frontend");
      }
      mKeyDownMap[id] = character >= 'a' && character <= 'z' ? character + 'A' - 'a' : character; // mKeys stores letters in upper case
      entry.configRTTIIndex = configRTTIInfo::KEY_MAP_INDEX + id;
    } else {
      entry.configRTTIIndex = std::get<1>(key);
      entry.callback = std::move(std::get<2>(key));
    }
    keys += (keys.empty() ? "[" : " [") + name + "]";
  };
  [&]<size_t... I>(std::index_sequence<I...>) { (addKey(std::get<I>(argTuple)), ...); }(std::make_index_sequence<nKeys>());
  if constexpr (sizeof...(Args) == nKeys + 2) {
    keys = std::get<nKeys + 1>(argTuple);
  }
  mConfigRTTI->helpTextEntries.emplace_back(std::move(keys), std::get<nKeys>(argTuple));
}

const std::vector<std::string>& GPUDisplayFrontend::getHelpText() const { return mConfigRTTI->helpText; }

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
    const uint32_t base = k >= 256 ? k - SPECIAL_KEYS_SHIFTED : k; // A shifted special key is possible if the special key is, SHIFT itself is the modifier
    const bool possible = base != KEY_SHIFT && ((base < N_SPECIAL_KEYS && !rtti.keyNames[base].empty()) || (k < 256 && rtti.possibleKeys.find((char)k) != std::string_view::npos));
    if (rtti.configKeys[k].callback || rtti.isKeyMap(rtti.configKeys[k].configRTTIIndex)) {
      boundKeys.emplace_back(k);
    } else if (possible) {
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
  std::vector<std::string> mappedKeys(KEY_MAP_MAX);
  int32_t keyWidth = 0;
  for (uint16_t k : boundKeys) {
    keyWidth = std::max<int32_t>(keyWidth, rtti.getKeyName(k).size() + 2);
  }
  for (uint16_t k : boundKeys) {
    const uint32_t index = rtti.configKeys[k].configRTTIIndex;
    if (rtti.isKeyMap(index)) {
      mappedKeys[index - configRTTIInfo::KEY_MAP_INDEX] = "[" + rtti.getKeyName(k) + "]";
      GPUInfo("  %-*s %-28s %s", keyWidth, ("[" + rtti.getKeyName(k) + "]").c_str(), "(key down map)", rtti.keyMapNames[index - configRTTIInfo::KEY_MAP_INDEX].c_str());
    } else {
      const bool hasRTTI = index != configRTTIInfo::INVALID_INDEX;
      GPUInfo("  %-*s %-28s %s", keyWidth, ("[" + rtti.getKeyName(k) + "]").c_str(), hasRTTI ? rtti.rttiEntries[index].configStruct.c_str() : "(no config option)", hasRTTI ? rtti.rttiEntries[index].name.c_str() : "");
    }
  }
  GPUInfo("Key down mappings (id, name, key):");
  for (uint32_t i = 0; i < KEY_MAP_MAX; i++) {
    GPUInfo("  %2u %-28s %s", i, rtti.keyMapNames[i].c_str(), mappedKeys[i].c_str());
  }
  std::string available;
  for (uint16_t k : availableKeys) {
    available += (available.empty() ? "[" : " [") + rtti.getKeyName(k) + "]";
  }
  GPUInfo("Available keys: %s", available.c_str());
  GPUInfo("%zu options with RTTI, %u of them with a key, %zu bound keys, %zu available keys", rtti.rttiEntries.size(), nOptionsWithKey, boundKeys.size(), availableKeys.size());
}

#include "GPUDisplayFrontendConfigKeys.inc"
#include "GPUDisplayFrontendConfigInfo.inc"

void GPUDisplay::buildConfig()
{
  auto& rtti = mFrontend->mConfigRTTI;
  rtti = {new GPUDisplayFrontend::configRTTIInfo, [](GPUDisplayFrontend::configRTTIInfo* p) { delete p; }};

  rtti->keyNames.resize(rtti->configKeys.size());
  rtti->keyMapNames.resize(GPUDisplayFrontend::KEY_MAP_MAX + 1); // Including MAX
#define GPUDisplayFrontendAddKey(code, key) rtti->keyNames[code] = #key
#define GPUDisplayFrontendKeyDownMap(code, key) rtti->keyMapNames[code] = #key
#include "frontend/GPUDisplayFrontendKeys.h"

  std::tie(rtti->rttiEntries, rtti->configStructNames) = std::apply([this](auto... s) { return qConfigGenRTTI<GPUDisplayFrontend::configRTTIExt, std::remove_reference_t<decltype(this->*s)>...>(); }, rtti->configStructs);
  std::apply([&](auto... s) { (rtti->rttiReverseId.emplace_back(sizeof(this->*s), GPUDisplayFrontend::configRTTIInfo::INVALID_INDEX), ...); }, rtti->configStructs);
  for (uint32_t i = 0; i < rtti->rttiEntries.size(); i++) {
    const auto& entry = rtti->rttiEntries[i];
    rtti->rttiReverseId[entry.configStructIndex][entry.offset] = i;
  }

  buildConfigKeys();

  rtti->configInfos.resize(rtti->rttiEntries.size());
  for (uint32_t i = 0; i < rtti->rttiEntries.size(); i++) { // Info from the qConfig message, with the value as %s as in qconfig.cxx
    const auto& e = rtti->rttiEntries[i];
    if (e.message.empty()) {
      continue;
    }
    if (!rtti->valueString(*this, i)) {
      GPUError("No info message support for type %s of %s", e.type.c_str(), e.name.c_str());
      continue;
    }
    rtti->configInfos[i] = [this, i, message = e.message]() {
      char buffer[256];
      const int32_t n = snprintf(buffer, sizeof(buffer), message.c_str(), mFrontend->mConfigRTTI->valueString(*this, i)->c_str());
      if (n < 0 || n >= (int32_t)sizeof(buffer)) {
        GPUError("Info message too long: %s", message.c_str());
        return;
      }
      SetInfo("%s", buffer);
    };
  }
  buildConfigInfo();
  rtti->buildHelpText();
}

bool GPUDisplayFrontend::processKeyLambda(uint32_t key)
{
  if (key < N_SPECIAL_KEYS && mKeys[KEY_SHIFT]) {
    key += SPECIAL_KEYS_SHIFTED;
  }
  if (key < mConfigRTTI->configKeys.size() && mConfigRTTI->configKeys[key].callback) {
    mConfigRTTI->configKeys[key].callback();
    if (const uint32_t index = mConfigRTTI->configKeys[key].configRTTIIndex; index < mConfigRTTI->configInfos.size()) {
      showConfigInfo(index);
    }
    return true;
  }
  return false;
}

void GPUDisplayFrontend::showConfigInfo(uint32_t index)
{
  if (mConfigRTTI->configInfos[index]) {
    mConfigRTTI->configInfos[index]();
  } else if (const auto value = mConfigRTTI->valueString(*mDisplay, index)) { // Generic info for options without info callback
    const auto& e = mConfigRTTI->rttiEntries[index];
    mDisplay->SetInfo("%s.%s changed to %s", mConfigRTTI->configStructNames[e.configStructIndex].c_str(), e.name.c_str(), value->c_str());
  }
}

void GPUDisplay::HandleKey(uint8_t key)
{
  if (key == mFrontend->KEY_ALT) {
    mFrontend->mKeys[mFrontend->KEY_CTRL] = false; // Release CTRL with alt, to avoid orienting along y automatically!
  } else if (mFrontend->processKeyLambda(key)) {
  }
}
