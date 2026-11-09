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

/// \file GPUDisplayFrontendConfigConsole.inc
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDCONFIGCONSOLE_INC
#define GPUDISPLAYFRONTENDCONFIGCONSOLE_INC

void GPUDisplayFrontend::buildConsoleCommands()
{
  using args = std::span<const GPUDisplayConsole::consoleArg>;
  auto& commands = mConsole->mConsoleCommands;
  commands["help"] = {[this](args) { mConsole->printCommands([this](const std::string& line) { mConsole->consolePrint("%s", line.c_str()); }); }, "", "List available console commands"};
  commands["clearHistory"] = {[this](args) {
                                mConsole->mConsoleHistory.resize(1); // Keep the title
                                mConsole->mConsoleTextCurrent = mConsole->mScroll = 0;
                                mConsole->mConsoleTextPos = std::min<uint32_t>(mConsole->mConsoleTextPos, mConsole->mConsoleText.size()); // Shown text is mConsoleText again
                                mConsole->mConsoleInfoLastMessage.clear();
                              },
                              "", "Clear the console history"};
  commands["runScript"] = {[this](args a) { mConsole->runScriptFile(std::get<std::string>(a[0])); }, "s", "Execute script file"};

  auto writeFile = [this](const char* command, const std::string& file, const char* success, const std::function<void(std::ostream&)>& write) { // Writes the file with write, reports errors opening / writing, or success ("<success> <file>")
    std::ofstream out(file);
    if (!out) {
      mConsole->consolePrint("%s: cannot open file %s", command, file.c_str());
      return;
    }
    write(out);
    out.close();
    mConsole->consolePrint(out ? "%s %s" : "%s: error writing file %s", out ? success : command, file.c_str());
  };
  auto floatNumber = [](const GPUDisplayConsole::consoleArg& a) { return (float)std::get<double>(a); };
  auto parseKey = [this](const GPUDisplayConsole::consoleArg& a) -> int32_t { // Key code (+ SPECIAL_KEYS_SHIFTED with SHIFT) from a code, a character or a key name (SHIFT_...), -1 if invalid
    if (const int64_t* code = std::get_if<int64_t>(&a)) {
      return *code > 0 && *code < (int64_t)N_TOTAL_KEYS ? (int32_t)*code : -1;
    }
    const std::string* str = std::get_if<std::string>(&a);
    if (!str) { // Floating point number
      return -1;
    }
    const std::string& name = *str;
    if (name.size() == 1) {
      return (uint8_t)name[0];
    }
    const bool shift = name.starts_with("SHIFT_");
    const std::string keyName = shift ? name.substr(6) : name;
    if (keyName.empty()) { // Would match the codes without name
      return -1;
    }
    const auto& names = mConfiguration->keyNames;
    const auto it = std::find(names.begin(), names.end(), keyName);
    const int32_t key = it == names.end() ? -1 : it - names.begin();
    return key < 0 || !shift ? key : (key < (int32_t)N_SPECIAL_KEYS ? key + (int32_t)SPECIAL_KEYS_SHIFTED : -1);
  };
  commands["setXYZ"] = {[this, floatNumber](args a) {
                          const float xyz[3] = {floatNumber(a[0]), floatNumber(a[1]), floatNumber(a[2])};
                          const float angles[3] = {mDisplay->mAngle[0], mDisplay->mAngle[1], mDisplay->mAngle[2]};
                          mDisplay->setCamera(xyz, angles);
                        },
                        "ddd", "Set the camera position x y z"};
  commands["setCameraAngles"] = {[this, floatNumber](args a) {
                                   const float toRad = CAMath::Pi() / 180.f;
                                   const float xyz[3] = {mDisplay->mXYZ[0], mDisplay->mXYZ[1], mDisplay->mXYZ[2]};
                                   const float angles[3] = {floatNumber(a[1]) * toRad, floatNumber(a[0]) * toRad, floatNumber(a[2]) * toRad}; // As mAngle: pitch, yaw, roll
                                   mDisplay->setCamera(xyz, angles);
                                 },
                                 "ddd", "Set the camera direction: yaw pitch roll in degrees"};
  commands["injectKey"] = {[this, parseKey](args a) {
                             const int32_t key = parseKey(a[0]);
                             if (key < 0) {
                               mConsole->consolePrint("injectKey: invalid key, expected a key code, a character, or a key name");
                               return;
                             }
                             mDisplay->HandleExternalGUIKey(key, true); // Handled as key also while the console is shown
                           },
                           "a", "Press a key: key code, character (\"a\"), or key name (UP, F1, SHIFT_F1, ...)"};
  commands["exit"] = {[this](args) {
                        mDisplay->mCfgC.displayControl.store(2);
                        mConsole->mScriptQueue.clear(); // No further script lines
                        showConfigInfo(mConfiguration->getRTTIIndex(*mDisplay, &GPUDisplayControl::displayControl));
                      },
                      "", "Exit"};
  commands["nextEvent"] = {[this](args) {
                             mDisplay->mCfgC.displayControl.store(1);
                             mConsole->waitFor([this, n = mDisplay->mNEventsShown] { return mDisplay->mNEventsShown != n; });
                           },
                           "", "Show the next event, a script continues when it is shown"};
  commands["addAnimationPoint"] = {[this, floatNumber](args a) {
                                     const float toRad = CAMath::Pi() / 180.f;
                                     const float xyz[3] = {floatNumber(a[0]), floatNumber(a[1]), floatNumber(a[2])};
                                     const float angles[3] = {floatNumber(a[4]) * toRad, floatNumber(a[3]) * toRad, floatNumber(a[5]) * toRad}; // As mAngle: pitch, yaw, roll
                                     mDisplay->addAnimationPoint(xyz, angles);
                                     mDisplay->SetInfo("Added Animation point (%d points, %6.2f seconds)", (int32_t)mDisplay->mAnimationVectors.size(), mDisplay->mCfgC.animationPoints.values.back());
                                   },
                                   "dddddd", "Add an animation point: x y z yaw pitch roll (degrees)"};
  commands["startAnimation"] = {[this](args) { mDisplay->runAnimation(); }, "", "Run the animation"};
  commands["takeScreenshot"] = {[this](args a) {
                                  mDisplay->takeScreenshot();
                                  mDisplay->mScreenshotFile = std::get<std::string>(a[0]);
                                  mConsole->waitFor([this] { return !mDisplay->mCfgC.requestScreenshot; }); // Otherwise further script lines would change the settings or the file before the screenshot is taken
                                  showConfigInfo(mConfiguration->getRTTIIndex(*mDisplay, &GPUDisplayControl::requestScreenshot));
                                },
                                "s", "Take a screenshot to the given file, a script continues after it is taken"};
  auto unbind = [this](int32_t key) { // Removes the binding of the key, returns the name of the previous binding, empty if none
    const int32_t binding = mConfiguration->removeKeyBinding(key);
    if (GPUDisplayConfiguration::isKeyMap(binding)) {
      mKeyDownMap[binding - GPUDisplayConfiguration::KEY_MAP_INDEX] = 0;
    }
    return mConfiguration->bindingName(binding);
  };
  commands["bindKey"] = {[this, parseKey, unbind](args a) {
                           auto& rtti = *mConfiguration;
                           const std::string& name = std::get<std::string>(a[1]);
                           const uint32_t mapId = std::find(rtti.keyMapNames.begin(), rtti.keyMapNames.begin() + KEY_MAP_MAX, name) - rtti.keyMapNames.begin();
                           const auto action = rtti.keyActionsNames.find(name);
                           const int32_t key = parseKey(a[0]);
                           if (key < 0 || (mapId < KEY_MAP_MAX && key >= 256)) {
                             mConsole->consolePrint("bindKey: invalid key, expected a key code, a character, or a key name (SHIFT_ only for actions)");
                             return;
                           } else if (mapId < KEY_MAP_MAX && key >= 'A' && key <= 'Z') { // Held letters use the entry of the lower case letter, as add_key_binding
                             mConsole->consolePrint("bindKey: key down mappings need lower case letters, the upper case letter would be SHIFT + key");
                             return;
                           } else if (mapId >= KEY_MAP_MAX && action == rtti.keyActionsNames.end()) {
                             mConsole->consolePrint("bindKey: unknown action or key down mapping %s (listKeys)", name.c_str());
                             return;
                           }
                           if (const std::string previous = unbind(key); previous.size() && previous != name) {
                             mConsole->consolePrint("bindKey: key [%s] was bound to %s, now unbound from it", rtti.getKeyName(key).c_str(), previous.c_str());
                           }
                           if (mapId < KEY_MAP_MAX) {
                             if (const int32_t previousKey = rtti.currentMappedKeys[mapId]; previousKey >= 0) { // The previous key of this mapping
                               rtti.removeKeyBinding(previousKey);
                             }
                             rtti.setKeyBinding(key, GPUDisplayConfiguration::KEY_MAP_INDEX + mapId);
                             mKeyDownMap[mapId] = key >= 'a' && key <= 'z' ? key + 'A' - 'a' : key; // mKeys stores letters in upper case, as add_key_binding
                           } else {
                             rtti.setKeyBinding(key, action->second);
                           }
                         },
                         "as", "Bind a key (code, character, or name, SHIFT_ for special keys with SHIFT) to an action or a key down mapping (listKeys), replacing the binding of the key; a key down mapping has one key (letters in lower case), an action keeps its other keys"};
  commands["clearKeyBindings"] = {[this](args) {
                                    for (int32_t key : mConfiguration->currentMappedKeys) {
                                      if (key >= 0) {
                                        mConfiguration->removeKeyBinding(key);
                                      }
                                    }
                                    mKeyDownMap.fill(0);
                                  },
                                  "", "Remove all key down mappings (bindKey)"};
  commands["resetAnimation"] = {[this](args) {
                                  mDisplay->resetAnimation();
                                  mDisplay->SetInfo("Reset Animation points", 1);
                                },
                                "", "Remove all animation points"};
  auto pointIndex = [this](const char* command, const GPUDisplayConsole::consoleArg& a) -> std::optional<int32_t> { // Index of an animation point or -1, error message otherwise
    const int64_t i = std::get<int64_t>(a);
    if (i < -1 || i >= (int64_t)mDisplay->mAnimationVectors.size()) {
      mConsole->consolePrint("%s: no animation point %lld (%zu points)", command, (long long)i, mDisplay->mAnimationVectors.size());
      return std::nullopt;
    }
    return i;
  };
  commands["removeAnimationPoint"] = {[this, pointIndex](args a) {
                                        if (const auto i = pointIndex("removeAnimationPoint", a[0])) {
                                          mDisplay->removeAnimationPoint(*i);
                                        }
                                      },
                                      "i", "Remove the animation point with this index, -1: the selected one, or the last one if none is selected"};
  commands["selectAnimationPoint"] = {[this, pointIndex](args a) {
                                        if (const auto i = pointIndex("selectAnimationPoint", a[0])) {
                                          mDisplay->selectAnimationPoint(*i);
                                        }
                                      },
                                      "i", "Select an animation point (-1: none): loads its settings and camera, later changes of the light settings are stored in it"};
  auto fileCommand = [this](std::string GPUDisplayOptions::* file, void (GPUDisplay::*function)()) { // Sets the file name option (as in the GUI) and runs the function
    return [this, file, function](args a) {
      mDisplay->mCfgO.*file = std::get<std::string>(a[0]);
      (mDisplay->*function)();
    };
  };
  commands["saveCamera"] = {fileCommand(&GPUDisplayOptions::glPosFile, &GPUDisplay::SaveGLPos), "s", "Save the camera position to the file"};
  commands["loadCamera"] = {fileCommand(&GPUDisplayOptions::glPosFile, &GPUDisplay::LoadGLPos), "s", "Load the camera position from the file"};
  commands["saveAnimation"] = {fileCommand(&GPUDisplayOptions::glAnimationFile, &GPUDisplay::SaveGLAnimation), "s", "Save the animation path to the file"};
  commands["loadAnimation"] = {fileCommand(&GPUDisplayOptions::glAnimationFile, &GPUDisplay::LoadGLAnimation), "s", "Load the animation path from the file"};
  commands["moveCamera"] = {[this, floatNumber](args a) {
                              const float xyz[3] = {mDisplay->mXYZ[0] + floatNumber(a[0]), mDisplay->mXYZ[1] + floatNumber(a[1]), mDisplay->mXYZ[2] + floatNumber(a[2])};
                              const float angles[3] = {mDisplay->mAngle[0], mDisplay->mAngle[1], mDisplay->mAngle[2]};
                              mDisplay->setCamera(xyz, angles);
                            },
                            "ddd", "Move the camera by dx dy dz (global coordinates)"};
  commands["rotateCamera"] = {[this, floatNumber](args a) {
                                const float angles[3] = {floatNumber(a[0]), floatNumber(a[1]), floatNumber(a[2])};
                                mDisplay->rotateCamera(angles);
                              },
                              "ddd", "Rotate the camera relative to the view, as the keyboard: yaw (right), pitch (down), roll (left) in degrees"};
  commands["lookAt"] = {[this, floatNumber](args a) {
                          const float target[3] = {floatNumber(a[0]), floatNumber(a[1]), floatNumber(a[2])};
                          if (!mDisplay->lookAt(target)) {
                            mConsole->consolePrint("lookAt: the target is the camera position");
                          }
                        },
                        "ddd", "Turn the camera to the point x y z, keeping its position"};
  static constexpr std::string_view groups = "lhroc"; // Order of GPUDisplayConfiguration::configStructs
  auto optionName = [this](uint32_t i) { return std::string(1, groups[mConfiguration->rttiEntries[i].configStructIndex]) + "." + mConfiguration->rttiEntries[i].name; };
  auto findOptions = [this](const std::string& name, bool allowGroups) -> std::vector<uint32_t> { // Indices of the option l.name, with allowGroups also of the group l / h / r / o / c or all; empty if not found
    const size_t group = name.size() && (name.size() == 1 ? allowGroups : name[1] == '.') ? groups.find(name[0]) : std::string_view::npos;
    std::vector<uint32_t> retVal;
    for (uint32_t i = 0; i < mConfiguration->rttiEntries.size(); i++) {
      const auto& e = mConfiguration->rttiEntries[i];
      if ((allowGroups && name == "all") || (e.configStructIndex == group && (name.size() == 1 || e.name == name.substr(2)))) {
        retVal.emplace_back(i);
      }
    }
    return retVal;
  };
  auto controlState = [this](uint32_t i) { return mConfiguration->rttiEntries[i].configStructIndex == GPUDisplayConfiguration::configStructIndex<GPUDisplayControl>(); }; // Control state and actions (mCfgC): not saved, not reset with a group

  auto defaultValue = [this](uint32_t i) -> std::optional<GPUDisplayConsole::consoleArg> { // Default from the RTTI, as setValue expects it, nullopt if it is not a number (e.g. GPUDisplayMultiSliderValue)
    const auto& e = mConfiguration->rttiEntries[i];
    if (e.type == "std::string" || e.type == "bool") {
      return e.defaultValue;
    }
    int64_t intValue = 0;
    const char* end = e.defaultValue.c_str() + e.defaultValue.size();
    if (std::from_chars(e.defaultValue.c_str(), end, intValue).ptr == end) { // Integers exactly, not through double
      return intValue;
    }
    double doubleValue = 0.;
    if (const auto r = std::from_chars(e.defaultValue.c_str(), end, doubleValue); r.ec == std::errc() && r.ptr == end) {
      return doubleValue;
    }
    return std::nullopt;
  };
  auto toText = [](const GPUDisplayConsole::consoleArg& a) { // Argument as text
    return std::visit([](const auto& v) {
      if constexpr (std::is_same_v<std::remove_cvref_t<decltype(v)>, std::string>) {
        return v;
      } else {
        return qConfig::print_type(v);
      }
    },
                      a);
  };

  commands["set"] = {[this, findOptions](args a) {
                       const std::string& name = std::get<std::string>(a[0]);
                       const auto index = findOptions(name, false);
                       if (index.empty()) {
                         mConsole->consolePrint("set: unknown option %s, expected l. / h. / r. / o. / c. and the name", name.c_str());
                         return;
                       }
                       if (const std::string error = mConfiguration->setValue(*mDisplay, index[0], a[1]); error.size()) {
                         mConsole->consolePrint("set %s: %s", name.c_str(), error.c_str());
                       } // Quiet: no info text, no GUI setter
                     },
                     "sa", "Set an option: l. / h. / r. / o. / c. (light, heavy, renderer, options, control) and its name, value"};
  commands["print"] = {[this, findOptions, optionName](args a) {
                         const std::string& name = std::get<std::string>(a[0]);
                         const auto index = findOptions(name, true);
                         if (index.empty()) {
                           mConsole->consolePrint("print: unknown option %s, expected all, l / h / r / o / c, or l. / h. / r. / o. / c. and an option name", name.c_str());
                           return;
                         }
                         for (uint32_t i : index) {
                           const auto value = mConfiguration->valueString(*mDisplay, i);
                           mConsole->consolePrint("%s = %s", optionName(i).c_str(), value ? value->c_str() : "(type not supported)");
                         }
                       },
                       "s", "Print an option (l. / h. / r. / o. / c. and its name), a group (l / h / r / o / c), or all"};
  commands["reset"] = {[this, findOptions, controlState, defaultValue, optionName](args a) {
                         const std::string& name = std::get<std::string>(a[0]);
                         const auto index = findOptions(name, true);
                         if (index.empty()) {
                           mConsole->consolePrint("reset: unknown option %s, expected all, l / h / r / o / c, or l. / h. / r. / o. / c. and an option name", name.c_str());
                           return;
                         }
                         for (uint32_t i : index) {
                           if (controlState(i)) { // Transient, no settings
                             if (index.size() == 1) {
                               mConsole->consolePrint("reset %s: control state, not resettable", optionName(i).c_str());
                             }
                             continue;
                           }
                           const auto value = defaultValue(i);
                           if (const std::string error = value ? mConfiguration->setValue(*mDisplay, i, *value) : "default is not a number, not resettable"; error.size()) {
                             mConsole->consolePrint("reset %s: %s", optionName(i).c_str(), error.c_str());
                           } // Quiet as set
                         }
                       },
                       "s", "Reset an option, a group (l / h / r / o), or all to the defaults, not the control state c"};
  commands["describe"] = {[this, findOptions, optionName](args a) {
                            const auto index = findOptions(std::get<std::string>(a[0]), false);
                            if (index.empty()) {
                              mConsole->consolePrint("describe: unknown option %s", std::get<std::string>(a[0]).c_str());
                              return;
                            }
                            const auto& e = mConfiguration->rttiEntries[index[0]];
                            const auto value = mConfiguration->valueString(*mDisplay, index[0]);
                            std::string keys;
                            for (uint32_t k = 0; k < mConfiguration->configKeys.size(); k++) {
                              if (mConfiguration->keyRTTIIndex(k) == index[0]) {
                                keys += (keys.empty() ? "[" : " [") + mConfiguration->getKeyName(k) + "]";
                              }
                            }
                            const bool inGUI = std::any_of(mConfiguration->guiElements.begin(), mConfiguration->guiElements.end(), [&](const auto& g) { return g.rttiIndex == index[0]; });
                            mConsole->consolePrint("%s (%s): %s", optionName(index[0]).c_str(), e.type.c_str(), e.description.empty() ? "no description" : e.description.c_str());
                            mConsole->consolePrint("  value %s, default %s%s%s%s%s", value ? value->c_str() : "(type not supported)", e.defaultValue.c_str(), e.min.size() ? ", min " : "", e.min.c_str(), e.max.size() ? ", max " : "", e.max.c_str());
                            mConsole->consolePrint("  keys %s, %s", keys.empty() ? "none" : keys.c_str(), inGUI ? "in the Qt GUI" : "not in the Qt GUI");
                          },
                          "s", "Describe an option (l. / h. / r. / o. / c. and its name): type, description, value, default, range, keys, GUI"};
  commands["find"] = {[this, optionName](args a) {
                        auto lower = [](std::string s) {
                          std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
                          return s;
                        };
                        const std::string text = lower(std::get<std::string>(a[0]));
                        uint32_t n = 0;
                        for (uint32_t i = 0; i < mConfiguration->rttiEntries.size(); i++) {
                          const auto& e = mConfiguration->rttiEntries[i];
                          if (lower(e.name).find(text) != std::string::npos || lower(e.description).find(text) != std::string::npos) {
                            mConsole->consolePrint("%s: %s", optionName(i).c_str(), e.description.c_str());
                            n++;
                          }
                        }
                        mConsole->consolePrint("find: %u options", n);
                      },
                      "s", "Find options whose name or description contains the text (case insensitive)"};
  commands["saveSettings"] = {[this, controlState, optionName, writeFile](args a) {
                                writeFile("saveSettings", std::get<std::string>(a[0]), "Settings and camera saved to", [&](std::ostream& out) {
                                  for (uint32_t i = 0; i < mConfiguration->rttiEntries.size(); i++) {
                                    const auto value = mConfiguration->valueString(*mDisplay, i, true);
                                    if (!value || controlState(i)) {
                                      continue;
                                    } else if (value->find_first_of("\r\n") != std::string::npos) {
                                      mConsole->consolePrint("saveSettings: %s not saved, scripts cannot contain line breaks", optionName(i).c_str());
                                      continue;
                                    }
                                    const bool isString = mConfiguration->rttiEntries[i].type == "std::string";
                                    std::string text = value->empty() ? "\"\"" : "";
                                    for (size_t k = 0; k < value->size(); k++) { // Escaped: a single argument, and plain text (a string) for string options also if it looks like a number
                                      const char c = (*value)[k];
                                      text += (c == ' ' || c == '"' || c == '\'' || c == '\\' || (isString && k == 0) ? "\\" : "") + std::string(1, c);
                                    }
                                    out << "set " << optionName(i) << " " << text << "\n";
                                  }
                                  const float toDeg = 180.f / CAMath::Pi();
                                  out << std::setprecision(std::numeric_limits<float>::max_digits10); // Camera position exact, angles up to the conversion to degrees
                                  out << "setXYZ " << mDisplay->mXYZ[0] << " " << mDisplay->mXYZ[1] << " " << mDisplay->mXYZ[2] << "\n";
                                  out << "setCameraAngles " << mDisplay->mAngle[1] * toDeg << " " << mDisplay->mAngle[0] * toDeg << " " << mDisplay->mAngle[2] * toDeg << "\n";
                                });
                              },
                              "s", "Save all settings and the camera as console script (run it with runScript)"};
  commands["waitFrames"] = {[this](args a) { mConsole->waitFrames(std::max<int64_t>(std::get<int64_t>(a[0]), 0)); }, "i", "In a script: run the following lines after n frames"};
  commands["waitSeconds"] = {[this](args a) { mConsole->waitSeconds(std::get<double>(a[0])); }, "d", "In a script: run the following lines after t seconds"};
  commands["benchmark"] = {[this](args a) { mConsole->startBenchmark(std::max<int64_t>(std::get<int64_t>(a[0]), 1)); }, "i", "Measure the frame rate of the next n frames (FPS limit / VSync apply), a script continues afterwards"};
  commands["echo"] = {[this, toText](args a) {
                        std::string text;
                        for (const auto& arg : a) {
                          text += (text.empty() ? "" : " ") + toText(arg);
                        }
                        mConsole->consolePrint("%s", text.c_str());
                      },
                      "+", "Print the arguments"};
  commands["printCamera"] = {[this](args) {
                               const float toDeg = 180.f / CAMath::Pi();
                               mConsole->consolePrint("Camera: x %f y %f z %f, yaw %f pitch %f roll %f", mDisplay->mXYZ[0], mDisplay->mXYZ[1], mDisplay->mXYZ[2], mDisplay->mAngle[1] * toDeg, mDisplay->mAngle[0] * toDeg, mDisplay->mAngle[2] * toDeg);
                             },
                             "", "Print the camera position and direction (degrees)"};
  commands["printStatus"] = {[this](args) { mConsole->consolePrint("%s", mDisplay->mGUIInfoStatus.empty() ? "No status yet (o.printInfoText / o.printInfoTextGUI)" : mDisplay->mGUIInfoStatus.c_str()); }, "", "Print the status line (FPS, counts, camera), updated every second"};
  commands["unbindKey"] = {[this, parseKey, unbind](args a) {
                             const int32_t key = parseKey(a[0]);
                             if (key < 0) {
                               mConsole->consolePrint("unbindKey: invalid key, expected a key code, a character, or a key name");
                               return;
                             } else if (mConfiguration->configKeys[key] == GPUDisplayConfiguration::KEY_UNBOUND) {
                               mConsole->consolePrint("unbindKey: key [%s] is not bound", mConfiguration->getKeyName(key).c_str());
                               return;
                             }
                             unbind(key);
                           },
                           "a", "Remove the binding of a key (action or key down mapping)"};
  commands["listKeys"] = {[this](args) { listKeys([this](const std::string& line) { mConsole->consolePrint("%s", line.c_str()); }); }, "", "List the key bindings, and actions / key down mappings without key"};
  commands["printHelpText"] = {[this](args) {
                                 for (const auto& line : getHelpText()) {
                                   mConsole->consolePrint("%s", line.c_str());
                                 }
                               },
                               "", "Print the help text of the keys (as h), with the current key bindings"};
  commands["unbindAll"] = {[this](args) {
                             mConfiguration->unbindAllKeys();
                             mKeyDownMap.fill(0);
                           },
                           "", "Remove all key bindings (actions and key down mappings, also the key of the console)"};
  commands["saveKeyBindings"] = {[this, writeFile](args a) {
                                   writeFile("saveKeyBindings", std::get<std::string>(a[0]), "Key bindings saved to", [&](std::ostream& out) {
                                     out << "unbindAll\n";
                                     for (uint32_t k = 1; k < mConfiguration->configKeys.size(); k++) {
                                       if (const int32_t binding = mConfiguration->configKeys[k]; binding != GPUDisplayConfiguration::KEY_UNBOUND) {
                                         const std::string key = mConfiguration->getKeyName(k);
                                         out << "bindKey " << (key.size() == 1 ? "\\" : "") << key << " " << mConfiguration->bindingName(binding) << "\n"; // Characters escaped: plain text, also digits, space, quotes
                                       }
                                     }
                                   });
                                 },
                                 "s", "Save all key bindings as console script: unbindAll, then bindKey for each key (run it with runScript)"};
  commands["resetView"] = {[this](args) {
                             mDisplay->mCfgC.resetScene.store(true);
                             showConfigInfo(mConfiguration->getRTTIIndex(*mDisplay, &GPUDisplayControl::resetScene));
                           },
                           "", "Reset the view"};
  commands["history"] = {[this, writeFile](args a) {
                           writeFile("history", std::get<std::string>(a[0]), "History saved to", [&](std::ostream& out) {
                             for (const auto& line : mConsole->mConsoleHistory) { // Title, commands, and their output (indented)
                               out << line << "\n";
                             }
                           });
                         },
                         "s", "Save the console history (commands and output) to a file"};
}

#endif // GPUDISPLAYFRONTENDCONFIGCONSOLE_INC
