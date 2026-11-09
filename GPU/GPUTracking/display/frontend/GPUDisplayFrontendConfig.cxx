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
#include "GPUCommonMath.h"
#include "GPUDisplayGUIChannel.h"
#include "GPUDisplayConsole.h"
#define GPUSETTINGS_ALL
#include "GPUSettings.h"
#include "utils/qconfig_rtti.h"
#include "utils/qconfig_internal_helpers.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <array>
#include <cctype>
#include <fstream>
#include <functional>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace o2::gpu;
struct GPUDisplayFrontend::configRTTIExt : public qConfig::qConfigRTTIentry {
};

using configKeyCallback = std::function<void()>;

struct GPUDisplayFrontend::GPUDisplayConfiguration {
  static constexpr std::tuple configStructs{&GPUDisplay::mCfgL, &GPUDisplay::mCfgH, &GPUDisplay::mCfgR, &GPUDisplay::mCfgO, &GPUDisplay::mCfgC}; // Config structs with RTTI, as members of GPUDisplay

  std::vector<configRTTIExt> rttiEntries;                                                // qconfig RTTI of all options of configStructs (struct, name, type, offset, default, min / max, description, message), the index identifies an option
  std::vector<configKeyCallback> configInfos;                                            // SetInfo callback for an option, indexed like rttiEntries, called after the callback of a key bound to it
  std::vector<std::vector<uint32_t>> rttiReverseId;                                      // [configStructIndex][offset] -> index in rttiEntries, INVALID_INDEX if no member at that offset
  static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();        // No entry: option index (rttiEntries) or, for keyAction / guiElements, no option
  static constexpr int32_t KEY_UNBOUND = -1;                                             // configKeys entry of a key without binding
  static constexpr int32_t KEY_MAP_INDEX = 1 << 24;                                      // configKeys entry of keys mapped to mKeyDownMap[entry - KEY_MAP_INDEX], entries below are indices in keyActions
  static bool isKeyMap(int32_t binding) { return binding >= KEY_MAP_INDEX; }             // configKeys entry of a key down mapping
  using keyDownMapEntry = std::tuple<uint8_t, uint8_t>;                                  // KEY_DOWN_MAP: key, KEY_MAP_ id
  using keyLambdaEntry = std::tuple<uint16_t, std::string, uint32_t, configKeyCallback>; // KEY_LAMBDA / KEY_ACTION: key, actionName, index in rttiEntries (INVALID_INDEX for KEY_ACTION), callback
  struct keyAction {
    std::string actionName;     // Unique, UPPER_SNAKE, used by bindKey
    uint32_t rttiIndex;         // Option changed by the action, INVALID_INDEX if none
    configKeyCallback callback; // Lambda of KEY_LAMBDA / KEY_ACTION
  };
  std::vector<keyAction> keyActions;                                                       // Key functions of KEY_LAMBDA / KEY_ACTION, keys can be bound to them by name (bindKey)
  std::unordered_map<std::string, uint32_t> keyActionsNames;                               // actionName -> index in keyActions
  std::vector<std::string> configStructNames;                                              // Names of the configStructs, by configStructIndex
  std::vector<std::vector<int32_t>> currentActionKeys;                                     // Reverse lookup of configKeys: keys bound to each action, by index in keyActions, in the order of binding
  std::vector<int32_t> currentMappedKeys;                                                  // Reverse lookup of configKeys: key of each key down mapping, by id, -1 if none
  std::array<int32_t, N_TOTAL_KEYS> configKeys;                                            // By key code, special keys with SHIFT at code + SPECIAL_KEYS_SHIFTED: index in keyActions, KEY_MAP_INDEX + id, or KEY_UNBOUND (filled in buildConfig)
  std::vector<std::string> keyNames;                                                       // Names of special keys (KEY_*) by key code < N_SPECIAL_KEYS, empty for other codes; for getKeyName and key names in console commands
  std::vector<std::string> keyMapNames;                                                    // Names of the key down mapping ids (KEY_MAP_*)
  std::vector<std::tuple<std::string, std::string, std::vector<int32_t>>> helpTextEntries; // Keys, text, and bindings (configKeys entries) of each add_key_binding call
  std::vector<std::string> helpText;                                                       // Aligned lines from helpTextEntries, by buildHelpText; use getHelpText, which rebuilds them if needed
  bool helpTextUpToDate = false;                                                           // Set by buildHelpText, reset when a key binding changes: getHelpText calls rebuildHelpText
  struct guiElementCallbacks {
    uint32_t rttiIndex;                                 // Option of the element
    std::function<void(const GPUDisplayGUIValue&)> set; // GUI_LAMBDA setter, applies a value from the GUI
  };
  std::vector<guiElementCallbacks> guiElements;                                                                               // By element id
  std::vector<GPUDisplayGUIChannel::Frame> guiLayout;                                                                         // Frames of the add_gui_binding calls
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
  std::string setValue(GPUDisplay& display, uint32_t index, const GPUDisplayConsole::consoleArg& value);   // Converted to the type of the option, empty string on success, otherwise the error
  std::optional<std::string> valueString(GPUDisplay& display, uint32_t index, bool precise = false) const; // Current value of an option, printed with print_type as in qconfig.cxx, nullopt for unsupported types; precise: floats with all digits, to read them back exactly
  template <class T, class M>
  uint32_t getRTTIIndex(GPUDisplay& display, M T::* member) // RTTI index of a direct member of a config struct
  {
    const T& obj = configStruct<T>(display);
    return rttiReverseId[configStructIndex<T>()][(const char*)&(obj.*member) - (const char*)&obj];
  }
  void buildHelpText() // Keys left-aligned, the ':' aligned after the longest keys
  {
    size_t width = 0;
    for (const auto& [keys, text, bindings] : helpTextEntries) {
      width = std::max(width, keys.size());
    }
    helpText.clear();
    for (const auto& [keys, text, bindings] : helpTextEntries) {
      helpText.emplace_back(keys + std::string(width - keys.size(), ' ') + ": " + text);
    }
    helpTextUpToDate = true;
  }
  void rebuildHelpText() // Keys of each entry from the current bindings, in the order of add_key_binding: [N/A] if unbound, [a|b] for several keys; entries without bindings (mouse) keep their text
  {
    for (auto& [keys, text, bindings] : helpTextEntries) {
      if (bindings.empty()) {
        continue;
      }
      keys.clear();
      for (int32_t binding : bindings) {
        std::string bound;
        std::vector<int32_t> keyCodes = boundKeys(binding);
        std::sort(keyCodes.begin(), keyCodes.end()); // By key code, independent of the order of binding
        for (int32_t k : keyCodes) {
          bound += (bound.empty() ? "" : "|") + getKeyName(k);
        }
        keys += (keys.empty() ? "[" : " [") + (bound.empty() ? "N/A" : bound) + "]";
      }
    }
    buildHelpText();
  }
  std::vector<int32_t> boundKeys(int32_t binding) const // Keys bound to the action or key down mapping
  {
    if (isKeyMap(binding)) {
      const int32_t key = currentMappedKeys[binding - KEY_MAP_INDEX];
      return key < 0 ? std::vector<int32_t>() : std::vector<int32_t>{key};
    }
    return currentActionKeys[binding];
  }
  void setKeyBinding(uint32_t key, int32_t binding) // Binds the unbound key, also in the reverse lookups (mKeyDownMap is set by the caller)
  {
    configKeys[key] = binding;
    if (isKeyMap(binding)) {
      currentMappedKeys[binding - KEY_MAP_INDEX] = key;
    } else {
      currentActionKeys[binding].emplace_back(key);
    }
    helpTextUpToDate = false;
  }
  int32_t removeKeyBinding(uint32_t key) // Unbinds the key, also in the reverse lookups (mKeyDownMap is reset by the caller), returns the previous binding
  {
    const int32_t binding = configKeys[key];
    if (isKeyMap(binding)) {
      currentMappedKeys[binding - KEY_MAP_INDEX] = -1;
    } else if (binding != KEY_UNBOUND) {
      std::erase(currentActionKeys[binding], key);
    }
    configKeys[key] = KEY_UNBOUND;
    helpTextUpToDate = false;
    return binding;
  }
  void unbindAllKeys() // Unbinds all keys, also in the reverse lookups (mKeyDownMap is reset by the caller)
  {
    configKeys.fill(KEY_UNBOUND);
    currentMappedKeys.assign(KEY_MAP_MAX, -1);
    for (auto& keys : currentActionKeys) {
      keys.clear();
    }
    helpTextUpToDate = false;
  }
  std::string bindingName(int32_t binding) const // Name of the action or key down mapping, empty if unbound
  {
    return isKeyMap(binding) ? keyMapNames[binding - KEY_MAP_INDEX] : (binding >= 0 ? keyActions[binding].actionName : "");
  }
  uint32_t keyRTTIIndex(uint32_t key) const // Option changed by the action bound to the key, INVALID_INDEX if none
  {
    const int32_t binding = configKeys[key];
    return binding != KEY_UNBOUND && !isKeyMap(binding) ? keyActions[binding].rttiIndex : INVALID_INDEX;
  }
  std::string getKeyName(uint16_t key) const // Name of a special key (KEY_*), the character otherwise, the code if neither, SHIFT_ for shifted special keys
  {
    if (key >= 256) {
      return "SHIFT_" + getKeyName(key - SPECIAL_KEYS_SHIFTED);
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
  const bool changedH = memcmp((void*)&oldCfgH, (void*)&mCfgH, sizeof(mCfgH)) != 0, changedL = memcmp((void*)&oldCfgL, (void*)&mCfgL, sizeof(mCfgL)) != 0, changedR = memcmp((void*)&oldCfgR, (void*)&mCfgR, sizeof(mCfgR)) != 0;
  if (mAnimate && (changedH || changedL || changedR || mCfgC.resetScene.load())) {
    StopAnimation();
  }
  if (changedH) {
    mUpdateEventData = true;
  }
  if (mCfgL.propagateTracks != 0 && !mTracksArePropagated) {
    mUpdateVertexLists = true;
  }
  if (changedL || changedR) {
    mUpdateDrawCommands = true;
  }
  if (oldCfgR.drawQualityMSAA != mCfgR.drawQualityMSAA || oldCfgR.drawQualityDownsampleFSAA != mCfgR.drawQualityDownsampleFSAA || oldCfgL.depthBuffer != mCfgL.depthBuffer || oldCfgR.screenshotScaleFactor != mCfgR.screenshotScaleFactor) {
    mUpdateRenderPipeline = true;
  }
  if (oldCfgL.depthBuffer != mCfgL.depthBuffer) { // Also for set / scripts, not only key and GUI
    mBackend->setDepthBuffer();
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
  if (oldCfgH.trackFilter != mCfgH.trackFilter) {
    mUpdateTrackFilter = true;
  }
  mCfgO.rotateModel |= mCfgO.rotateAroundZ;
  if (mCfgC.showConsole != mConsoleShown) {
    mConsoleShown = mCfgC.showConsole;
    const double start = mConsoleTimer.IsRunning() ? std::max(0., mConfig.internal.consoleDelay - mConsoleTimer.GetCurrentElapsedTime()) : 0.; // Reverse a running animation from its current position
    mConsoleTimer.Reset();
    mConsoleTimer.AddTime(start);
    mConsoleTimer.Start();
    if (mCfgC.showConsole) {
      const bool shift = mFrontend->mKeys[mFrontend->KEY_SHIFT]; // Kept while the console is shown (scrolling, shifted keys), frontends still update it
      std::fill(std::begin(mFrontend->mKeys), std::end(mFrontend->mKeys), false);
      mFrontend->mKeys[mFrontend->KEY_SHIFT] = shift;
    }
  }
  if (mCfgO.applyToAllPoints && oldCfgL.animationMode != mCfgL.animationMode) { // Also for keys; the animation's own changes are in mOldCfg, so this is only user input
    for (auto& p : mAnimationVectors) {
      p.cfgL.animationMode = mCfgL.animationMode;
    }
  }
  if (mCfgC.animationPoints.selected >= 0 && mCfgC.animationPoints.selected < (int32_t)mAnimationVectors.size()) { // Update animation settings of selected point
    mAnimationVectors[mCfgC.animationPoints.selected].cfgL = mCfgL;
  }
  updateAnimationPointsGUI();
}

void GPUDisplay::HandleExternalGUIKey(int32_t key, bool bypassConsole)
{
  bool shifted = key >= 'A' && key <= 'Z';
  if (key >= 256) { // Special key with SHIFT, from translateExternalKey
    shifted = true;
    key -= mFrontend->SPECIAL_KEYS_SHIFTED;
  }
  bool oldShift = mFrontend->mKeys[mFrontend->KEY_SHIFT];
  mFrontend->mKeys[mFrontend->KEY_SHIFT] = shifted;
  HandleKey(key, true, bypassConsole);
  mFrontend->mKeys[mFrontend->KEY_SHIFT] = oldShift;
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
  constexpr size_t nKeys = (0 + ... + (std::is_same_v<std::remove_cvref_t<Args>, GPUDisplayConfiguration::keyLambdaEntry> || std::is_same_v<std::remove_cvref_t<Args>, GPUDisplayConfiguration::keyDownMapEntry>));
  static_assert(nKeys >= 1 && (sizeof...(Args) == nKeys + 1 || sizeof...(Args) == nKeys + 2), "add_key_binding needs at least one key, the help text, and optionally the keys to show in the help text");
  auto argTuple = std::forward_as_tuple(args...);
  std::string keys;
  std::vector<int32_t> bindings;
  auto addKey = [&](auto& key) {
    const uint16_t character = std::get<0>(key);
    const std::string name = mConfiguration->getKeyName(character);
    int32_t binding;
    if (mConfiguration->configKeys[character] != GPUDisplayConfiguration::KEY_UNBOUND) {
      throw std::runtime_error("Key [" + name + "] assigned twice in display frontend");
    }
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(key)>, GPUDisplayConfiguration::keyDownMapEntry>) {
      const uint8_t id = std::get<1>(key);
      if (mKeyDownMap[id]) {
        throw std::runtime_error("Key down mapping " + mConfiguration->keyMapNames[id] + " assigned twice in display frontend");
      }
      mKeyDownMap[id] = character >= 'a' && character <= 'z' ? character + 'A' - 'a' : character; // mKeys stores letters in upper case
      binding = GPUDisplayConfiguration::KEY_MAP_INDEX + id;
    } else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(key)>, GPUDisplayConfiguration::keyLambdaEntry>) {
      const std::string& actionName = std::get<1>(key);
      const auto& mapNames = mConfiguration->keyMapNames;
      if (mConfiguration->keyActionsNames.contains(actionName) || std::find(mapNames.begin(), mapNames.end(), actionName) != mapNames.end()) {
        throw std::runtime_error("Key action " + actionName + " defined twice in display frontend, or name of a key down mapping");
      }
      binding = mConfiguration->keyActions.size();
      mConfiguration->keyActionsNames.emplace(actionName, binding);
      mConfiguration->keyActions.push_back({actionName, std::get<2>(key), std::move(std::get<3>(key))});
      mConfiguration->currentActionKeys.emplace_back();
    } else {
      static_assert(false, "invalid code path");
    }
    mConfiguration->setKeyBinding(character, binding);
    bindings.emplace_back(binding);
    keys += (keys.empty() ? "[" : " [") + name + "]";
  };
  [&]<size_t... I>(std::index_sequence<I...>) { (addKey(std::get<I>(argTuple)), ...); }(std::make_index_sequence<nKeys>());
  if constexpr (sizeof...(Args) == nKeys + 2) {
    keys = std::get<nKeys + 1>(argTuple);
  }
  mConfiguration->helpTextEntries.emplace_back(std::move(keys), std::get<nKeys>(argTuple), std::move(bindings));
}

struct guiType { // Element type and choices, GUI_CHECKBOX, GUI_DROPDOWN(...), etc. in GPUDisplayFrontendConfigQtGUI.inc
  guiType(GPUDisplayGUIChannel::ElementType t, std::vector<std::pair<std::string, GPUDisplayGUIValue>> c = {}, std::function<GPUDisplayGUIChannel::Options()> o = {}, uint32_t l = 0, int64_t b = 0, double s = 0) : type(t), choices(std::move(c)), options(std::move(o)), lines(l), bitMask(b), step(s) {}
  GPUDisplayGUIChannel::ElementType type;
  std::vector<std::pair<std::string, GPUDisplayGUIValue>> choices;
  std::function<GPUDisplayGUIChannel::Options()> options; // GUI_SLIDER_DYNAMIC / GUI_DROPDOWN_DYNAMIC
  uint32_t lines = 0;                                     // GUI_TEXTEDIT
  int64_t bitMask = 0;                                    // GUI_CHECKBOX_BIT
  double step = 0;                                        // GUI_SLIDER_STEP / GUI_SLIDER_DYNAMIC_STEP / GUI_MULTI_SLIDER
};

static std::vector<std::pair<std::string, GPUDisplayGUIValue>> guiSliderValues(std::initializer_list<GPUDisplayGUIValue> values) // Choices of GUI_SLIDER_VALUES, named by their value
{
  auto name = [](const auto& x) -> std::string {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, std::string>) {
      return x;
    } else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, GPUDisplayMultiSliderValue>) {
      return "";
    } else {
      return qConfig::print_type(x);
    }
  };
  std::vector<std::pair<std::string, GPUDisplayGUIValue>> retVal;
  for (const auto& v : values) {
    retVal.emplace_back(std::visit(name, v), v);
  }
  return retVal;
}

template <typename T>
static std::vector<std::pair<std::string, GPUDisplayGUIValue>> guiNamedChoices(const T& names) // Choices of GUI_DROPDOWN_NAMES
{
  std::vector<std::pair<std::string, GPUDisplayGUIValue>> retVal;
  for (const auto& name : names) {
    retVal.emplace_back(name, (int64_t)retVal.size());
  }
  return retVal;
}

static GPUDisplayGUIChannel::Options guiRange(double min, double max) { return {.min = min, .max = max, .choices = {}}; } // For the lambda of GUI_SLIDER_DYNAMIC

struct guiElement { // GUI_LAMBDA(...)
  guiElement(std::string n, uint32_t i, guiType t, std::function<void(const GPUDisplayGUIValue&)> s, std::function<GPUDisplayGUIValue(const GPUDisplayGUIChannel::Settings&)> g) : name(std::move(n)), rttiIndex(i), type(std::move(t)), set(std::move(s)), get(std::move(g)) {}
  std::string name;
  uint32_t rttiIndex;
  guiType type;
  std::function<void(const GPUDisplayGUIValue&)> set;
  std::function<GPUDisplayGUIValue(const GPUDisplayGUIChannel::Settings&)> get;
};

template <class T>
static void guiAssign(T& dst, const GPUDisplayGUIValue& value) // Default GUI_LAMBDA setter: assign the GUI value, converted to the type of the option
{
  std::visit([&](const auto& v) {
    using V = std::remove_cvref_t<decltype(v)>;
    if constexpr (std::is_same_v<T, V>) {
      dst = v;
    } else if constexpr (std::is_same_v<T, GPUDisplayMultiSliderValue> || std::is_same_v<V, GPUDisplayMultiSliderValue>) {
      throw std::runtime_error("GUI value of a multi slider for another element type, or vice versa");
    } else if constexpr (std::is_same_v<T, std::string>) {
      if constexpr (std::is_same_v<V, std::string>) {
        dst = v;
      } else {
        dst = std::to_string(v);
      }
    } else if constexpr (std::is_same_v<V, std::string>) {
      dst = static_cast<T>(std::stod(v)); // Throws for invalid text
    } else {
      dst = static_cast<T>(v);
    }
  },
             value);
}

template <class T>
static GPUDisplayGUIValue toGUIValue(const T& v)
{
  if constexpr (std::is_same_v<T, std::string>) {
    return v;
  } else if constexpr (std::is_integral_v<T>) {
    return (int64_t)v;
  } else {
    static_assert(std::is_floating_point_v<T>, "Unsupported option type for the Qt GUI");
    return (double)v;
  }
}

static GPUDisplayGUIValue toGUIValue(const GPUDisplayMultiSliderValue& v) { return v; }

template <class T>
static GPUDisplayGUIValue toGUIValue(const GPUDisplayCopyableAtomic<T>& v)
{
  return toGUIValue(v.load());
}

struct guiVertical { // Argument between the elements of add_gui_binding: alignment of the following elements
  bool vertical;
};

template <typename... Args>
void GPUDisplayFrontend::add_gui_binding(Args&&... args)
{
  constexpr size_t nElements = (0 + ... + std::is_same_v<std::remove_cvref_t<Args>, guiElement>);
  constexpr size_t nHorizontal = (0 + ... + std::is_same_v<std::remove_cvref_t<Args>, bool>);
  constexpr size_t nVertical = (0 + ... + std::is_same_v<std::remove_cvref_t<Args>, guiVertical>);
  static_assert(nElements >= 1 && nHorizontal <= 1 && sizeof...(Args) <= nElements + nHorizontal + nVertical + 1, "add_gui_binding needs at least one GUI element, and optionally the description, the horizontal flag, and guiVertical between elements");
  GPUDisplayGUIChannel::Frame frame;
  bool horizontal = false; // Initially the flag of the frame, changed by guiVertical
  auto getHorizontal = [&](auto& arg) {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(arg)>, bool>) {
      horizontal = arg;
    }
  };
  (getHorizontal(args), ...);
  auto addElement = [&](auto& element) {
    using Type = GPUDisplayGUIChannel::ElementType;
    if (element.type.bitMask) { // The element sets / shows only these bits of the option
      element.set = [this, set = element.set, get = element.get, mask = element.type.bitMask](const GPUDisplayGUIValue& v) {
        const int64_t current = std::get<int64_t>(get({mDisplay->mCfgH, mDisplay->mCfgL, mDisplay->mCfgR, mDisplay->mCfgO, mDisplay->mCfgC}));
        set(std::get<int64_t>(v) ? (current | mask) : (current & ~mask));
      };
      element.get = [get = element.get, mask = element.type.bitMask](const GPUDisplayGUIChannel::Settings& cfg) -> GPUDisplayGUIValue { return (int64_t)((std::get<int64_t>(get(cfg)) & mask) != 0); };
    }
    static const GPUDisplayGUIChannel::Settings defaults; // Only for the types of the values
    const uint32_t id = mConfiguration->guiElements.size();
    auto& el = frame.elements.emplace_back(GPUDisplayGUIChannel::Element{.name = element.name, .description = element.rttiIndex < mConfiguration->rttiEntries.size() ? mConfiguration->rttiEntries[element.rttiIndex].description : "", .rttiIndex = element.rttiIndex, .type = element.type.type, .id = id, .isInteger = std::holds_alternative<int64_t>(element.get(defaults)), .isString = std::holds_alternative<std::string>(element.get(defaults)), .lines = element.type.lines, .step = element.type.step, .choices = std::move(element.type.choices), .dynamicOptions = std::move(element.type.options), .get = element.get});
    if (el.dynamicOptions && !((el.type == Type::SLIDER && el.choices.empty()) || el.type == Type::DROPDOWN)) {
      throw std::runtime_error("GUI element " + element.name + ": dynamic options only for sliders with range and dropdown menus");
    }
    if (el.type == Type::SLIDER && !el.dynamicOptions) {
      if (element.rttiIndex >= mConfiguration->rttiEntries.size()) {
        throw std::runtime_error("GUI slider " + element.name + " needs an option with RTTI for min and max values");
      }
      const auto& e = mConfiguration->rttiEntries[element.rttiIndex];
      if (e.min.empty() || e.max.empty()) {
        throw std::runtime_error("GUI slider " + element.name + ": option " + e.name + " needs min and max values");
      }
      el.min = std::stod(e.min);
      el.max = std::stod(e.max);
    }
    if (el.type == Type::TEXTBOX && element.rttiIndex < mConfiguration->rttiEntries.size() && !mConfiguration->rttiEntries[element.rttiIndex].min.empty() && !mConfiguration->rttiEntries[element.rttiIndex].max.empty()) {
      el.min = std::stod(mConfiguration->rttiEntries[element.rttiIndex].min);
      el.max = std::stod(mConfiguration->rttiEntries[element.rttiIndex].max);
      el.hasRange = true;
    }
    if (el.step > 0 && !(el.type == Type::MULTI_SLIDER || (el.type == Type::SLIDER && el.choices.empty()))) {
      throw std::runtime_error("GUI element " + element.name + ": a step only for sliders with a range and multi sliders");
    }
    if (el.type == Type::MULTI_SLIDER && (!std::holds_alternative<GPUDisplayMultiSliderValue>(element.get(defaults)) || el.step <= 0)) {
      throw std::runtime_error("GUI multi slider " + element.name + " needs an option of type GPUDisplayMultiSliderValue and a step > 0");
    }
    if ((el.type == Type::RADIO_BUTTONS || el.type == Type::DROPDOWN) ? (el.choices.empty() && !el.dynamicOptions) : (!el.choices.empty() && el.type != Type::SLIDER)) {
      throw std::runtime_error("GUI element " + element.name + ": choices are needed for radio buttons and dropdown menus, and only allowed for them and sliders");
    }
    el.horizontal = horizontal;
    mConfiguration->guiElements.emplace_back(GPUDisplayConfiguration::guiElementCallbacks{element.rttiIndex, std::move(element.set)});
  };
  auto addArg = [&](auto& arg) {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(arg)>, guiElement>) {
      addElement(arg);
    } else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(arg)>, guiVertical>) {
      horizontal = !arg.vertical;
    } else if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(arg)>, bool>) {
      frame.description = arg;
    }
  };
  (addArg(args), ...);
  mConfiguration->guiLayout.emplace_back(std::move(frame));
}

void GPUDisplayFrontend::setGUILayout(GPUDisplayGUIChannel& channel) const { channel.layout = mConfiguration->guiLayout; }

void GPUDisplayFrontend::applyGUIValues(const GPUDisplayGUIUpdate& update)
{
  for (const auto& [id, value] : update.values) {
    if (id >= mConfiguration->guiElements.size()) {
      continue;
    }
    const auto& e = mConfiguration->guiElements[id];
    try {
      e.set(value);
    } catch (const std::exception& ex) {
      GPUError("Invalid value from the GUI for element %u: %s", id, ex.what());
      continue;
    }
    if (mDisplay->mCfgO.guiPrintInfo && e.rttiIndex < mConfiguration->configInfos.size()) {
      showConfigInfo(e.rttiIndex);
    }
  }
}

const std::vector<std::string>& GPUDisplayFrontend::getHelpText()
{
  if (!mConfiguration->helpTextUpToDate) {
    mConfiguration->rebuildHelpText();
  }
  return mConfiguration->helpText;
}

static std::pair<int32_t, uint16_t> keyOrder(uint16_t k) // 0-9, a-z, A-Z, other characters, special keys, special keys with SHIFT
{
  return {k >= '0' && k <= '9'   ? 0
          : k >= 'a' && k <= 'z' ? 1
          : k >= 'A' && k <= 'Z' ? 2
          : k >= 256             ? 5
          : k > 32               ? 3
                                 : 4,
          k};
}

void GPUDisplayFrontend::listKeys(const std::function<void(const std::string&)>& print)
{
  const auto& rtti = *mConfiguration;
  std::vector<uint16_t> boundKeys;
  for (uint32_t k = 1; k < rtti.configKeys.size(); k++) {
    if (rtti.configKeys[k] != GPUDisplayConfiguration::KEY_UNBOUND) {
      boundKeys.emplace_back(k);
    }
  }
  std::sort(boundKeys.begin(), boundKeys.end(), [](uint16_t a, uint16_t b) { return keyOrder(a) < keyOrder(b); });
  std::vector<std::pair<std::string, int32_t>> rows; // Key, binding
  for (uint16_t k : boundKeys) {
    rows.emplace_back("[" + rtti.getKeyName(k) + "]", rtti.configKeys[k]);
  }
  auto addUnbound = [&](int32_t binding) {
    if (rtti.boundKeys(binding).empty()) {
      rows.emplace_back("[N/A]", binding);
    }
  };
  for (uint32_t i = 0; i < rtti.keyActions.size(); i++) {
    addUnbound(i);
  }
  for (uint32_t i = 0; i < KEY_MAP_MAX; i++) {
    addUnbound(GPUDisplayConfiguration::KEY_MAP_INDEX + i);
  }
  size_t keyWidth = 0, nameWidth = 0;
  for (const auto& [key, binding] : rows) {
    keyWidth = std::max(keyWidth, key.size());
    nameWidth = std::max(nameWidth, rtti.bindingName(binding).size());
  }
  print("Key bindings (key, action or key down mapping, option):");
  for (const auto& [key, binding] : rows) {
    const std::string name = rtti.bindingName(binding);
    const uint32_t index = GPUDisplayConfiguration::isKeyMap(binding) ? GPUDisplayConfiguration::INVALID_INDEX : rtti.keyActions[binding].rttiIndex;
    const std::string option = GPUDisplayConfiguration::isKeyMap(binding) ? "(key down map)" : (index == GPUDisplayConfiguration::INVALID_INDEX ? "(no config option)" : rtti.rttiEntries[index].configStruct + " " + rtti.rttiEntries[index].name);
    print("  " + key + std::string(keyWidth - key.size() + 1, ' ') + name + std::string(nameWidth - name.size() + 1, ' ') + option);
  }
  print("");
  print("Key down mappings (id, name, key):");
  for (uint32_t i = 0; i < KEY_MAP_MAX; i++) {
    char line[128];
    snprintf(line, sizeof(line), "  %2u %-28s %s", i, rtti.keyMapNames[i].c_str(), rtti.currentMappedKeys[i] < 0 ? "[N/A]" : ("[" + rtti.getKeyName(rtti.currentMappedKeys[i]) + "]").c_str());
    print(line);
  }
}

void GPUDisplayFrontend::listConfig()
{
  const auto& rtti = *mConfiguration;
  std::vector<uint16_t> boundKeys, availableKeys;
  for (uint32_t k = 1; k < rtti.configKeys.size(); k++) {
    const uint32_t base = k >= 256 ? k - SPECIAL_KEYS_SHIFTED : k; // A shifted special key is possible if the special key is, SHIFT itself is the modifier
    const bool possible = base != KEY_SHIFT && ((base < N_SPECIAL_KEYS && !rtti.keyNames[base].empty()) || (k < 256 && rtti.possibleKeys.find((char)k) != std::string_view::npos));
    if (rtti.configKeys[k] != GPUDisplayConfiguration::KEY_UNBOUND) {
      boundKeys.emplace_back(k);
    } else if (possible) {
      availableKeys.emplace_back(k);
    }
  }
  std::sort(boundKeys.begin(), boundKeys.end(), [&](uint16_t a, uint16_t b) { return keyOrder(a) < keyOrder(b); });
  std::sort(availableKeys.begin(), availableKeys.end(), [&](uint16_t a, uint16_t b) { return keyOrder(a) < keyOrder(b); });

  GPUInfo("");
  GPUInfo("Display config options with RTTI (struct, option, type, default, in Qt GUI, keys):");
  uint32_t nOptionsWithKey = 0, nOptionsInGUI = 0;
  int32_t structWidth = 0, nameWidth = 0, typeWidth = 0, defaultWidth = 0; // Columns aligned to the longest entry
  for (const auto& e : rtti.rttiEntries) {
    structWidth = std::max<int32_t>(structWidth, e.configStruct.size());
    nameWidth = std::max<int32_t>(nameWidth, e.name.size());
    typeWidth = std::max<int32_t>(typeWidth, e.type.size());
    defaultWidth = std::max<int32_t>(defaultWidth, e.defaultValue.size());
  }
  for (uint32_t i = 0; i < rtti.rttiEntries.size(); i++) {
    const auto& e = rtti.rttiEntries[i];
    std::string keys;
    for (uint16_t k : boundKeys) {
      if (rtti.keyRTTIIndex(k) == i) {
        keys += (keys.empty() ? "[" : " [") + rtti.getKeyName(k) + "]";
      }
    }
    const bool inGUI = std::any_of(rtti.guiElements.begin(), rtti.guiElements.end(), [i](const auto& g) { return g.rttiIndex == i; });
    nOptionsWithKey += !keys.empty();
    nOptionsInGUI += inGUI;
    GPUInfo("  %-*s %-*s %-*s %-*s %-3s %s", structWidth, e.configStruct.c_str(), nameWidth, e.name.c_str(), typeWidth, e.type.c_str(), defaultWidth, e.defaultValue.c_str(), inGUI ? "GUI" : "---", keys.c_str());
  }
  GPUInfo("");
  listKeys([](const std::string& line) { GPUInfo("%s", line.c_str()); });
  GPUInfo("");
  static constexpr std::string_view availablePrefix = "Available keys: ";
  static constexpr size_t MAX_LINE_LENGTH = 240;
  std::string available(availablePrefix);
  for (uint16_t k : availableKeys) {
    const std::string key = "[" + rtti.getKeyName(k) + "]";
    if (available.size() > availablePrefix.size() && available.size() + 1 + key.size() > MAX_LINE_LENGTH) { // Wrap, without splitting a key
      GPUInfo("%s", available.c_str());
      available.assign(availablePrefix.size(), ' ');
    } else if (available.size() > availablePrefix.size()) {
      available += ' ';
    }
    available += key;
  }
  GPUInfo("%s", available.c_str());
  GPUInfo("");
  GPUInfo("Help text:");
  for (const auto& line : getHelpText()) {
    GPUInfo("  %s", line.c_str());
  }
  GPUInfo("");
  mConsole->printCommands([](const std::string& line) { GPUInfo("%s", line.c_str()); });
  GPUInfo("");
  GPUInfo("%zu options with RTTI, %u of them with a key, %u in the Qt GUI, %zu bound keys, %zu available keys, %zu console commands", rtti.rttiEntries.size(), nOptionsWithKey, nOptionsInGUI, boundKeys.size(), availableKeys.size(), mConsole->mConsoleCommands.size());
  GPUInfo("");
}

#include "GPUDisplayFrontendConfigKeys.inc"
#include "GPUDisplayFrontendConfigInfo.inc"
#include "GPUDisplayFrontendConfigQtGUI.inc"
#include "GPUDisplayFrontendConfigConsole.inc"

void GPUDisplay::buildConfig()
{
  auto& rtti = mFrontend->mConfiguration;
  rtti = {new GPUDisplayFrontend::GPUDisplayConfiguration, [](GPUDisplayFrontend::GPUDisplayConfiguration* p) { delete p; }};

  rtti->unbindAllKeys();
  mFrontend->mKeyDownMap.fill(0); // Set again by buildConfigKeys, also when the display is started again
  rtti->keyNames.resize(rtti->configKeys.size());
  rtti->keyMapNames.resize(GPUDisplayFrontend::KEY_MAP_MAX + 1); // Including MAX
#define GPUDisplayFrontendAddKey(code, key) rtti->keyNames[code] = #key
#define GPUDisplayFrontendKeyDownMap(code, key) rtti->keyMapNames[code] = #key
#include "frontend/GPUDisplayFrontendKeys.h"

  std::tie(rtti->rttiEntries, rtti->configStructNames) = std::apply([this](auto... s) { return qConfigGenRTTI<GPUDisplayFrontend::configRTTIExt, std::remove_reference_t<decltype(this->*s)>...>(); }, rtti->configStructs);
  std::apply([&](auto... s) { (rtti->rttiReverseId.emplace_back(sizeof(this->*s), GPUDisplayFrontend::GPUDisplayConfiguration::INVALID_INDEX), ...); }, rtti->configStructs);
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
      const int32_t n = snprintf(buffer, sizeof(buffer), message.c_str(), mFrontend->mConfiguration->valueString(*this, i)->c_str());
      if (n < 0 || n >= (int32_t)sizeof(buffer)) {
        GPUError("Info message too long: %s", message.c_str());
        return;
      }
      SetInfo("%s", buffer);
    };
  }
  buildConfigInfo();
  buildConfigQtGUI();
  mFrontend->buildConsoleCommands();
  mFrontend->mConsole->mPrintToTerminal = mConfig.internal.consolePrintToTerminal;
  rtti->buildHelpText();
}

uint32_t GPUDisplayFrontend::shiftedKey(uint32_t key) const { return key < N_SPECIAL_KEYS && mKeys[KEY_SHIFT] ? key + SPECIAL_KEYS_SHIFTED : key; }

bool GPUDisplayFrontend::processKeyLambda(uint32_t key)
{
  key = shiftedKey(key);
  const int32_t binding = key < mConfiguration->configKeys.size() ? mConfiguration->configKeys[key] : GPUDisplayConfiguration::KEY_UNBOUND;
  if (binding != GPUDisplayConfiguration::KEY_UNBOUND && !GPUDisplayConfiguration::isKeyMap(binding)) {
    const auto& action = mConfiguration->keyActions[binding];
    action.callback();
    if (action.rttiIndex < mConfiguration->configInfos.size()) {
      showConfigInfo(action.rttiIndex);
    }
    return true;
  }
  return false;
}

void GPUDisplayFrontend::showConfigInfo(uint32_t index)
{
  if (mConfiguration->configInfos[index]) {
    mConfiguration->configInfos[index]();
  } else if (const auto value = mConfiguration->valueString(*mDisplay, index)) { // Generic info for options without info callback
    const auto& e = mConfiguration->rttiEntries[index];
    mDisplay->SetInfo("%s.%s changed to %s", mConfiguration->configStructNames[e.configStructIndex].c_str(), e.name.c_str(), value->c_str());
  }
}

void GPUDisplay::HandleKey(uint8_t key, bool external, bool bypassConsole)
{
  if (mCfgC.showConsole && !bypassConsole && mFrontend->mConfiguration->keyRTTIIndex(mFrontend->shiftedKey(key)) != mFrontend->mConfiguration->getRTTIIndex(*this, &GPUDisplayControl::showConsole)) { // All keys except the one toggling the console go to the console
    if (!external) {
      mFrontend->mConsole->handleConsoleKey(key, mFrontend->mKeys[mFrontend->KEY_SHIFT]);
    }
    return;
  }
  const uint8_t originKey = mFrontend->mKeyDownMap[mFrontend->KEY_MAP_FOCUS_ORIGIN], upKey = mFrontend->mKeyDownMap[mFrontend->KEY_MAP_FOCUS_ORIENT_UP];
  if (originKey && upKey && (key >= 'a' && key <= 'z' ? key + 'A' - 'a' : key) == originKey) { // mKeyDownMap has letters in upper case
    mFrontend->mKeys[upKey] = false;                                                           // Release the orient up key (CTRL) with the focus origin key (ALT), to avoid orienting along y automatically!
  }
  mFrontend->processKeyLambda(key);
}

std::string GPUDisplayFrontend::GPUDisplayConfiguration::setValue(GPUDisplay& display, uint32_t index, const GPUDisplayConsole::consoleArg& value)
{
  const auto& e = rttiEntries[index];
  char* ptr = configStructPtr(display, e.configStructIndex) + e.offset;
  if (e.type == "std::string") {
    const std::string* str = std::get_if<std::string>(&value);
    if (!str) {
      return "string expected";
    }
    *(std::string*)ptr = *str;
    return "";
  }
  if (e.isVector || e.arraySize) {
    return "unsupported type " + e.type;
  }
  if (e.type == "float") { // Floating point: also from an integer
    const double v = std::holds_alternative<int64_t>(value) ? (double)std::get<int64_t>(value) : (std::holds_alternative<double>(value) ? std::get<double>(value) : std::nan(""));
    if (std::isnan(v)) {
      return "number expected";
    } else if ((e.min.size() && v < std::stod(e.min)) || (e.max.size() && v > std::stod(e.max))) {
      return "value out of range [" + e.min + ", " + e.max + "]";
    }
    *(float*)ptr = v;
    return "";
  }
  int64_t v; // Integer types and bool, exactly, not through double
  if (const std::string* str = std::get_if<std::string>(&value)) {
    if (e.type != "bool" || (*str != "true" && *str != "false")) {
      return "number expected";
    }
    v = *str == "true";
  } else if (const int64_t* i = std::get_if<int64_t>(&value)) {
    v = *i;
  } else {
    return "integer expected";
  }
  if ((e.min.size() && v < std::stoll(e.min)) || (e.max.size() && v > std::stoll(e.max))) {
    return "value out of range [" + e.min + ", " + e.max + "]";
  }
  auto store = [&]<class T>(auto&& assign) -> std::string { // Within the range of the type T
    if (v < (int64_t)std::numeric_limits<T>::min() || (v > 0 && (uint64_t)v > (uint64_t)std::numeric_limits<T>::max())) {
      return "value out of range of " + e.type;
    }
    assign((T)v);
    return "";
  };
  if (e.type == "bool") {
    return store.template operator()<bool>([&](bool x) { *(bool*)ptr = x; });
  } else if (e.type == "int8_t") {
    return store.template operator()<int8_t>([&](int8_t x) { *(int8_t*)ptr = x; });
  } else if (e.type == "uint8_t") {
    return store.template operator()<uint8_t>([&](uint8_t x) { *(uint8_t*)ptr = x; });
  } else if (e.type == "int32_t") {
    return store.template operator()<int32_t>([&](int32_t x) { *(int32_t*)ptr = x; });
  } else if (e.type == "uint32_t") {
    return store.template operator()<uint32_t>([&](uint32_t x) { *(uint32_t*)ptr = x; });
  } else if (e.type == "int64_t") {
    return store.template operator()<int64_t>([&](int64_t x) { *(int64_t*)ptr = x; });
  } else if (e.type == "uint64_t") {
    return store.template operator()<uint64_t>([&](uint64_t x) { *(uint64_t*)ptr = x; });
  } else if (e.type == "GPUDisplayCopyableAtomic<int32_t>") {
    return store.template operator()<int32_t>([&](int32_t x) { ((GPUDisplayCopyableAtomic<int32_t>*)ptr)->store(x); });
  } else if (e.type == "GPUDisplayCopyableAtomic<bool>") {
    return store.template operator()<bool>([&](bool x) { ((GPUDisplayCopyableAtomic<bool>*)ptr)->store(x); });
  }
  return "unsupported type " + e.type;
}

std::optional<std::string> GPUDisplayFrontend::GPUDisplayConfiguration::valueString(GPUDisplay& display, uint32_t index, bool precise) const
{
  const auto& e = rttiEntries[index];
  const char* ptr = configStructPtr(display, e.configStructIndex) + e.offset;
  if (e.isVector || e.arraySize) {
    return std::nullopt;
  } else if (e.type == "bool") {
    return qConfig::print_type(*(const bool*)ptr);
  } else if (e.type == "float") {
    if (precise) {
      std::ostringstream s;
      s << std::setprecision(std::numeric_limits<float>::max_digits10) << *(const float*)ptr;
      return s.str();
    }
    return qConfig::print_type(*(const float*)ptr);
  } else if (e.type == "int8_t") {
    return qConfig::print_type(*(const int8_t*)ptr);
  } else if (e.type == "uint8_t") {
    return qConfig::print_type(*(const uint8_t*)ptr);
  } else if (e.type == "int32_t") {
    return qConfig::print_type(*(const int32_t*)ptr);
  } else if (e.type == "uint32_t") {
    return qConfig::print_type(*(const uint32_t*)ptr);
  } else if (e.type == "int64_t") {
    return qConfig::print_type(*(const int64_t*)ptr);
  } else if (e.type == "uint64_t") {
    return qConfig::print_type(*(const uint64_t*)ptr);
  } else if (e.type == "GPUDisplayCopyableAtomic<int32_t>") {
    return qConfig::print_type(((const GPUDisplayCopyableAtomic<int32_t>*)ptr)->load());
  } else if (e.type == "GPUDisplayCopyableAtomic<bool>") {
    return qConfig::print_type(((const GPUDisplayCopyableAtomic<bool>*)ptr)->load());
  } else if (e.type == "std::string") {
    return *(const std::string*)ptr;
  }
  return std::nullopt;
}
