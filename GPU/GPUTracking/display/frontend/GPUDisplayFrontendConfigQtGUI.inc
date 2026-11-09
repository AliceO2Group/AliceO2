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

/// \file GPUDisplayFrontendConfigQtGUI.inc
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDCONFIGQTGUI_INC
#define GPUDISPLAYFRONTENDCONFIGQTGUI_INC

#define GUI_CHECKBOX guiType(GPUDisplayGUIChannel::ElementType::CHECKBOX)
#define GUI_CHECKBOX_BIT(mask) guiType(GPUDisplayGUIChannel::ElementType::CHECKBOX, {}, {}, 0, mask) // Checkbox for these bits of an integer option
#define GUI_TEXTBOX guiType(GPUDisplayGUIChannel::ElementType::TEXTBOX)
#define GUI_TEXTEDIT(lines) guiType(GPUDisplayGUIChannel::ElementType::TEXTEDIT, {}, {}, lines)                   // Multi-line text edit, initially that many lines, resizeable with the mouse, applied when it loses the focus
#define GUI_SLIDER guiType(GPUDisplayGUIChannel::ElementType::SLIDER)                                             // Range from the min / max of the option
#define GUI_SLIDER_STEP(step) guiType(GPUDisplayGUIChannel::ElementType::SLIDER, {}, {}, 0, 0, step)              // Range from the min / max of the option, positions multiples of step from min
#define GUI_SLIDER_VALUES(...) guiType(GPUDisplayGUIChannel::ElementType::SLIDER, guiSliderValues({__VA_ARGS__})) // Slider with only these values
#define GUI_RADIO_BUTTONS(...) guiType(GPUDisplayGUIChannel::ElementType::RADIO_BUTTONS, {__VA_ARGS__})           // Choices {"name", value}, ...
#define GUI_DROPDOWN(...) guiType(GPUDisplayGUIChannel::ElementType::DROPDOWN, {__VA_ARGS__})
#define GUI_DROPDOWN_NAMES(names) guiType(GPUDisplayGUIChannel::ElementType::DROPDOWN, guiNamedChoices(names))             // Choices with these names, the value is the index
#define GUI_SLIDER_DYNAMIC(...) guiType(GPUDisplayGUIChannel::ElementType::SLIDER, {}, __VA_ARGS__)                        // Range from a lambda returning guiRange(min, max), called by the GUI after mUpdateGUIOptions while the display waits
#define GUI_SLIDER_DYNAMIC_STEP(step, ...) guiType(GPUDisplayGUIChannel::ElementType::SLIDER, {}, __VA_ARGS__, 0, 0, step) // As GUI_SLIDER_DYNAMIC, positions multiples of step from min
#define GUI_DROPDOWN_DYNAMIC(...) guiType(GPUDisplayGUIChannel::ElementType::DROPDOWN, {}, __VA_ARGS__)                    // Choices from a lambda returning GPUDisplayGUIChannel::Options{.choices = ...}, as GUI_SLIDER_DYNAMIC
#define GUI_MULTI_SLIDER(step) guiType(GPUDisplayGUIChannel::ElementType::MULTI_SLIDER, {}, {}, 0, 0, step)                // Several handles for a GPUDisplayMultiSliderValue: positions multiples of step, at least step apart, range 0 to its max; sent when a handle is released or clicked (selection)
#define GUI_BUTTON guiType(GPUDisplayGUIChannel::ElementType::BUTTON)                                                      // Body of GUI_LAMBDA runs when the button is clicked
// GUI element for an option, type one of the GUI_* above. The optional body sets the value from the GUI: val is a reference to obj.value, guiVal the GPUDisplayGUIValue from the GUI (int64_t, double, std::string, or GPUDisplayMultiSliderValue). Without body, guiVal is assigned to val.
#define GUI_LAMBDA(name, obj, value, type, ...) guiElement(std::string(name), mFrontend->mConfiguration->getRTTIIndex(*this, &std::remove_cvref_t<decltype(obj)>::value), type, [this](const GPUDisplayGUIValue& guiVal) { [[maybe_unused]] auto& val = obj.value; __VA_OPT__(if constexpr (true) { __VA_ARGS__ } else) guiAssign(val, guiVal); }, [](const GPUDisplayGUIChannel::Settings& cfg) { return toGUIValue(cfg.obj.value); })
#define GUI_ACTION(name, ...) guiElement(std::string(name), GPUDisplayFrontend::GPUDisplayConfiguration::INVALID_INDEX, GUI_BUTTON, [this](const GPUDisplayGUIValue&) { __VA_ARGS__ }, [](const GPUDisplayGUIChannel::Settings&) { return GPUDisplayGUIValue((int64_t)0); }) // Button without config option: runs the body

void GPUDisplay::buildConfigQtGUI()
{
  mFrontend->add_gui_binding(
    GUI_LAMBDA("Clusters", mCfgL, drawClusters, GUI_CHECKBOX),
    GUI_LAMBDA("Initial links", mCfgL, drawInitLinks, GUI_CHECKBOX),
    GUI_LAMBDA("Links", mCfgL, drawLinks, GUI_CHECKBOX),
    GUI_LAMBDA("Seeds", mCfgL, drawSeeds, GUI_CHECKBOX),
    GUI_LAMBDA("Tracklets", mCfgL, drawTracklets, GUI_CHECKBOX),
    GUI_LAMBDA("Sector tracks", mCfgL, drawTracks, GUI_CHECKBOX),
    GUI_LAMBDA("Extrapolated tracks", mCfgL, drawExtrapolatedTracks, GUI_CHECKBOX),
    GUI_LAMBDA("Final tracks", mCfgL, drawFinal, GUI_CHECKBOX),
    "Draw Types", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("TPC", mCfgL, drawTPC, GUI_CHECKBOX),
    GUI_LAMBDA("TRD", mCfgL, drawTRD, GUI_CHECKBOX),
    GUI_LAMBDA("TOF", mCfgL, drawTOF, GUI_CHECKBOX),
    GUI_LAMBDA("ITS", mCfgL, drawITS, GUI_CHECKBOX),
    "Detectors", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Look at origin", mCfgR, camLookOrigin, GUI_CHECKBOX, guiAssign(val, guiVal); mCfgR.cameraMode = val + 2 * mCfgR.camYUp;),
    GUI_LAMBDA("Toggle y upward", mCfgR, camYUp, GUI_CHECKBOX, guiAssign(val, guiVal); mCfgR.cameraMode = mCfgR.camLookOrigin + 2 * val;),
    GUI_LAMBDA("Rotate Model", mCfgO, rotateModel, GUI_CHECKBOX),
    GUI_LAMBDA("Rotate around Beamline", mCfgO, rotateAroundZ, GUI_CHECKBOX),
    GUI_LAMBDA("Project onto XY", mCfgH, projectXY, GUI_CHECKBOX),
    "Camera Options", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Line width", mCfgL, lineWidth, GUI_SLIDER),
    GUI_LAMBDA("Point size", mCfgL, pointSize, GUI_SLIDER),
    GUI_LAMBDA("FOV", mCfgR, fov, GUI_SLIDER));

  mFrontend->add_gui_binding(
    GUI_LAMBDA("MSAA", mCfgR, drawQualityMSAA, GUI_SLIDER_VALUES(0, 2, 4, 8, 16)),
    GUI_LAMBDA("FSAA", mCfgR, drawQualityDownsampleFSAA, GUI_SLIDER_VALUES(0, 2, 3, 4)),
    "Anti-Aliasing");

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Smooth points", mCfgL, smoothPoints, GUI_CHECKBOX),
    GUI_LAMBDA("Smooth lines", mCfgL, smoothLines, GUI_CHECKBOX),
    GUI_LAMBDA("Depth buffer", mCfgL, depthBuffer, GUI_CHECKBOX),
    GUI_LAMBDA("VSync", mCfgR, drawQualityVSync, GUI_CHECKBOX),
    GUI_LAMBDA("FPS limit", mCfgR, fpsLimit, GUI_DROPDOWN({"None", 0}, {"Refresh rate", 1}, {"fpsRate", 2})),
    GUI_LAMBDA("fpsRate", mCfgR, fpsRate, GUI_TEXTBOX),
    GUI_LAMBDA("GL Indirect Draw", mCfgR, useGLIndirectDraw, GUI_CHECKBOX),
    "Draw Quality", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Invert Colors", mCfgL, invertColors, GUI_CHECKBOX),
    GUI_LAMBDA("Draw Grid", mCfgL, drawGrid, GUI_CHECKBOX),
    "Colors / Grid", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Maximized", mCfgR, maximized, GUI_CHECKBOX),
    GUI_LAMBDA("Full Screen", mCfgR, fullScreen, GUI_CHECKBOX),
    true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Sector", mCfgL, drawSector, GUI_SLIDER), // -1 = all sectors
    GUI_LAMBDA("Separate Sectors", mCfgH, xzAdd, GUI_SLIDER, guiAssign(val, guiVal);),
    guiVertical{false},
    GUI_LAMBDA("Draw related sectors", mCfgL, drawRelatedSectors, GUI_CHECKBOX),
    GUI_LAMBDA("Separate extrapolated tracks", mCfgH, separateExtrapolatedTracks, GUI_CHECKBOX),
    GUI_LAMBDA("Split CE tracks", mCfgH, splitCETracks, GUI_CHECKBOX),
    "TPC Sectors");

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Exclude tracking step clusters", mCfgL, excludeClusters, GUI_DROPDOWN({"No", 0}, {"Clusters and tracks", 1}, {"Clusters, keep tracks", 2})),
    GUI_LAMBDA("Exclude rejected tracks", mCfgH, hideRejectedTracks, GUI_CHECKBOX),
    GUI_LAMBDA("Exclude rejected clusters", mCfgH, hideRejectedClusters, GUI_CHECKBOX),
    GUI_LAMBDA("Unmatched Clusters", mCfgH, hideUnmatchedClusters, GUI_CHECKBOX),
    "Exclude", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("splitPad", mCfgH, markClusters, GUI_CHECKBOX_BIT(0x1)),
    GUI_LAMBDA("splitTime", mCfgH, markClusters, GUI_CHECKBOX_BIT(0x2)),
    GUI_LAMBDA("edge", mCfgH, markClusters, GUI_CHECKBOX_BIT(0x4)),
    GUI_LAMBDA("singlePad", mCfgH, markClusters, GUI_CHECKBOX_BIT(0x8)),
    GUI_LAMBDA("rejectDistance", mCfgH, markClusters, GUI_CHECKBOX_BIT(0x10)),
    GUI_LAMBDA("rejectErr", mCfgH, markClusters, GUI_CHECKBOX_BIT(0x20)),
    GUI_LAMBDA("fakeAttached", mCfgH, markFakeClusters, GUI_CHECKBOX),
    GUI_LAMBDA("adjacent", mCfgH, markAdjacentClusters, GUI_CHECKBOX, val = std::get<int64_t>(guiVal) ? 31 : 0;), // All adjacent categories
    GUI_LAMBDA("Auto-color track clusters", mCfgL, colorClusters, GUI_CHECKBOX),
    "Mark Clusters", true);

  std::vector<std::pair<std::string, GPUDisplayGUIValue>> filterChoices = {{"0: No Filter", (int64_t)0}};
  for (uint32_t i = 0; i < mConfig.filterMacros.size(); i++) {
    std::string name = mConfig.filterMacros[i];
    std::erase_if(name, [](char c) { return c == '\n' || c == '\r'; });
    filterChoices.emplace_back(std::to_string(i + 1) + ": " + name.substr(0, 16), (int64_t)(i + 1));
  }
  filterChoices.emplace_back(std::to_string(mConfig.filterMacros.size() + 1) + ": Custom Macro", (int64_t)(mConfig.filterMacros.size() + 1));
  mFrontend->add_gui_binding(
    GUI_LAMBDA("TPC", mCfgH, drawTPCTracks, GUI_CHECKBOX),
    GUI_LAMBDA("TRD", mCfgH, drawTRDTracks, GUI_CHECKBOX),
    GUI_LAMBDA("TOF", mCfgH, drawTOFTracks, GUI_CHECKBOX),
    GUI_LAMBDA("ITS", mCfgH, drawITSTracks, GUI_CHECKBOX),
    GUI_LAMBDA("", mCfgH, drawTracksAndFilter, GUI_RADIO_BUTTONS({"AND", 1}, {"OR", 0})),
    GUI_LAMBDA("Filter Macro", mCfgH, trackFilter, guiType(GPUDisplayGUIChannel::ElementType::DROPDOWN, filterChoices)),
    GUI_LAMBDA("Custom Macro", mCfgO, filterMacroCustom, GUI_TEXTEDIT(1), { // ROOT macro code, reload and apply the custom filter if selected, the change is not visible in the config structs
      guiAssign(val, guiVal);
      if ((uint32_t)mCfgH.trackFilter == mConfig.filterMacros.size() + 1) {
        mUpdateTrackFilter = true;
        mUpdateEventData = true;
      }
    }),
    "Track Filters", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Tracks", mCfgL, propagateTracks, GUI_DROPDOWN({"Hits connected", 0}, {"Propagated to vertex", 1}, {"Propagated inwards and outwards", 2}, {"Monte Carlo track", 3})),
    GUI_LAMBDA("Loopers", mCfgH, propagateLoopers, GUI_CHECKBOX),
    "Extrapolation", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Collision", mCfgL, showCollision, GUI_SLIDER_DYNAMIC([this]() { return guiRange(-1, mNCollissions - 1); })), // -1 = all collisions
    GUI_LAMBDA("Color-code collisions", mCfgL, colorCollisions, GUI_CHECKBOX),
    "Collisions");

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Show clusters Only", mCfgH, clustersOnly, GUI_CHECKBOX),
    GUI_LAMBDA("Clusters on nominal row x", mCfgH, clustersOnNominalRow, GUI_CHECKBOX),
    "Tracking Debug", true);

  mFrontend->add_gui_binding(
    GUI_ACTION("Save Camera", SaveGLPos();),
    GUI_ACTION("Load Camera", LoadGLPos();),
    GUI_ACTION("Save Animation", SaveGLAnimation();),
    GUI_ACTION("Load Animation", LoadGLAnimation();),
    GUI_LAMBDA("Camera File", mCfgO, glPosFile, GUI_TEXTBOX),
    GUI_LAMBDA("Animation File", mCfgO, glAnimationFile, GUI_TEXTBOX),
    "Load / Save", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Screenshot", mCfgC, requestScreenshot, GUI_BUTTON, takeScreenshot();),
    GUI_ACTION("Animation", runAnimation();),
    GUI_LAMBDA("Record Animation", mCfgO, animateScreenshot, GUI_DROPDOWN({"no", 0}, {"to screenshots", 1})),
    GUI_LAMBDA("Filename", mCfgO, mscreenShotName, GUI_TEXTBOX),
    guiVertical{true},
    GUI_LAMBDA("Scale Factor", mCfgR, screenshotScaleFactor, GUI_SLIDER),
    GUI_LAMBDA("Animation Delay", mCfgO, animationDelay, GUI_SLIDER_DYNAMIC_STEP(0.05, []() { return guiRange(0.05f, 20.f); })),
    guiVertical{false},
    GUI_ACTION("Add Point", setAnimationPoint(); SetInfo("Added Animation point (%d points, %6.2f seconds)", (int32_t)mAnimationVectors.size(), mCfgC.animationPoints.values.back());),
    GUI_ACTION("Remove Point", removeAnimationPoint(); SetInfo("Removed Animation point", 1);),
    GUI_ACTION("Reset Animation", resetAnimation(); SetInfo("Reset Animation points", 1);),
    GUI_LAMBDA("Animation Mode", mCfgL, animationMode, GUI_DROPDOWN_NAMES(ANIMATION_MODE_NAMES_SHORT), guiAssign(val, guiVal); if (!mCfgO.guiPrintInfo) { SetInfo("Animation mode set to %s", ANIMATION_MODE_NAMES_LONG[val].c_str()); }), // clang-format off
    GUI_LAMBDA("Apply to all", mCfgO, applyToAllPoints, GUI_CHECKBOX),
    guiVertical{true},
    GUI_LAMBDA("Points", mCfgC, animationPoints, GUI_MULTI_SLIDER(0.05), setAnimationPointsFromGUI(std::get<GPUDisplayMultiSliderValue>(guiVal));), // clang-format on
    "Screenshots / Animation", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Next Event", mCfgC, displayControl, GUI_BUTTON, val.store(1);),
    GUI_LAMBDA("Exit", mCfgC, displayControl, GUI_BUTTON, val.store(2);),
    GUI_LAMBDA("Reset", mCfgC, resetScene, GUI_BUTTON, val.store(true);),
    GUI_LAMBDA("Reset GUI", mCfgO, guiScale, GUI_BUTTON, val = 1.f;),
    GUI_ACTION("Close GUI", mFrontend->stopGUI();),
    GUI_LAMBDA("SendInfoText", mCfgO, guiPrintInfo, GUI_CHECKBOX),
    GUI_LAMBDA("GUIInfo", mCfgO, printInfoTextGUI, GUI_CHECKBOX),
    GUI_LAMBDA("HelpText", mCfgO, printInfoTextAlways, GUI_CHECKBOX),
    GUI_LAMBDA("StatusText", mCfgO, printInfoText, GUI_CHECKBOX_BIT(1)), // Bit 0: status line on screen (bit 1: in the console)
    GUI_LAMBDA("DarkMode", mCfgO, guiDarkMode, GUI_CHECKBOX),
    true);
}

#endif // GPUDISPLAYFRONTENDCONFIGQTGUI_INC
