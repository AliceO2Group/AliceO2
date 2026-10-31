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
#define GUI_TEXTBOX guiType(GPUDisplayGUIChannel::ElementType::TEXTBOX)
#define GUI_SLIDER guiType(GPUDisplayGUIChannel::ElementType::SLIDER)                                             // Range from the min / max of the option
#define GUI_SLIDER_VALUES(...) guiType(GPUDisplayGUIChannel::ElementType::SLIDER, guiSliderValues({__VA_ARGS__})) // Slider with only these values
#define GUI_RADIO_BUTTONS(...) guiType(GPUDisplayGUIChannel::ElementType::RADIO_BUTTONS, {__VA_ARGS__})           // Choices {"name", value}, ...
#define GUI_DROPDOWN(...) guiType(GPUDisplayGUIChannel::ElementType::DROPDOWN, {__VA_ARGS__})
#define GUI_BUTTON guiType(GPUDisplayGUIChannel::ElementType::BUTTON) // Body of GUI_LAMBDA runs when the button is clicked
// GUI element for an option, type one of the GUI_* above. The optional body sets the value from the GUI: val is a reference to obj.value, guiVal the GPUDisplayGUIValue from the GUI (int64_t, double, or std::string). Without body, guiVal is assigned to val.
#define GUI_LAMBDA(name, obj, value, type, ...) guiElement(std::string(name), mFrontend->mConfigRTTI->getRTTIIndex(*this, &std::remove_cvref_t<decltype(obj)>::value), type, [this](const GPUDisplayGUIValue& guiVal) { [[maybe_unused]] auto& val = obj.value; __VA_OPT__(if constexpr (true) { __VA_ARGS__ } else) guiAssign(val, guiVal); }, [this]() { return toGUIValue(obj.value); })

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
    GUI_LAMBDA("Upward Orientation", mCfgR, camYUp, GUI_CHECKBOX, guiAssign(val, guiVal); mCfgR.cameraMode = mCfgR.camLookOrigin + 2 * val;),
    GUI_LAMBDA("Rotate Model", mCfgC, mRotateModel, GUI_CHECKBOX),
    GUI_LAMBDA("Rotate around Beamline", mCfgC, mRotateAroundZ, GUI_CHECKBOX),
    GUI_LAMBDA("Project onto XY", mCfgH, projectXY, GUI_CHECKBOX),
    "Camera Options", true);

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Line width", mCfgL, lineWidth, GUI_SLIDER),
    GUI_LAMBDA("Point size", mCfgL, pointSize, GUI_SLIDER));

  mFrontend->add_gui_binding(
    GUI_LAMBDA("MSAA", mCfgR, drawQualityMSAA, GUI_SLIDER_VALUES(0, 2, 4, 8)),
    GUI_LAMBDA("FSAA", mCfgR, drawQualityDownsampleFSAA, GUI_SLIDER_VALUES(0, 2, 3, 4)),
    "Anti-Aliasing");

  mFrontend->add_gui_binding(
    GUI_LAMBDA("Smooth points", mCfgL, smoothPoints, GUI_CHECKBOX),
    GUI_LAMBDA("Smooth lines", mCfgL, smoothLines, GUI_CHECKBOX),
    GUI_LAMBDA("Depth buffer", mCfgL, depthBuffer, GUI_CHECKBOX, guiAssign(val, guiVal); mBackend->setDepthBuffer();),
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
    GUI_LAMBDA("Next Event", mCfgC, mDisplayControl, GUI_BUTTON, val.store(1);),
    GUI_LAMBDA("Exit", mCfgC, mDisplayControl, GUI_BUTTON, val.store(2);),
    GUI_LAMBDA("Reset", mCfgC, mResetScene, GUI_BUTTON, val.store(true);),
    GUI_LAMBDA("Reset GUI", mCfgC, mGuiScale, GUI_BUTTON, val = 1.f;),
    GUI_LAMBDA("Close GUI", mCfgC, dummy, GUI_BUTTON, mFrontend->stopGUI();),
    GUI_LAMBDA("InfoText", mCfgC, mGuiPrintInfo, GUI_CHECKBOX),
    GUI_LAMBDA("HelpText", mCfgC, mPrintInfoTextAlways, GUI_CHECKBOX),
    GUI_LAMBDA("StatusText", mCfgC, mPrintInfoText, GUI_CHECKBOX, val = std::get<int64_t>(guiVal) ? (val | 1) : (val & ~1);), // Bit 0: status line on screen (bit 1: in the console)
    true);
}

#endif // GPUDISPLAYFRONTENDCONFIGQTGUI_INC
