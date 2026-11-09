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

/// \file GPUSettingsListDisplay.h
/// \author David Rohr

// This file contains macros to generate all settings for the GPU Reconstruction.
// Macros are used in the following places:
// Create ConfigurableParam object for workflow.
// Configure standalone benchmark.
// Create plain-C struct for GPU code.
// Create static constexpr with default values for GPU run time compilation

#include "GPUSettingsTypes.h"

#if (!defined(__CLING__) && !defined(G__DICTIONARY)) || defined(BeginSubConfig) // File should not be included without defining the macros, but rootcling will do for dictionary generation

// clang-format off
BeginNamespace(o2::gpu)

// Light settings concerning the event display (can be changed without rebuilding vertices)
BeginSubConfig(GPUSettingsDisplayLight, light, configStandalone.display, "GLL", 'g', "Light OpenGL display settings", display_light)
AddOption(animationMode, int32_t, 0, "", 0, "", min(0), max(6))
AddOption(smoothPoints, bool, true, "", 0, "Apply smoothing to points")
AddOption(smoothLines, bool, false, "", 0, "Apply smoothing to lines")
AddOption(depthBuffer, bool, false, "", 0, "Enable Z-buffer")
AddOption(drawClusters, bool, true, "", 0, "Highlight clusters")
AddOption(drawLinks, bool, false, "", 0, "Highlight links")
AddOption(drawInitLinks, bool, false, "", 0, "Highlight cleaned-up links")
AddOption(drawSeeds, bool, false, "", 0, "Highlight seeds")
AddOption(drawTracklets, bool, false, "", 0, "Highlight tracklets")
AddOption(drawTracks, bool, false, "", 0, "Highlight sector tracks")
AddOption(drawExtrapolatedTracks, bool, false, "", 0, "Highlight global sector tracks prolonged into adjacent sector")
AddOption(drawFinal, bool, false, "", 0, "Highlight final tracks")
AddOption(excludeClusters, int32_t, 0, "", 0, "Exclude clusters from selected draw objects from display, (2 = exclude clusters but still show tracks)")
AddOption(drawSector, int32_t,  -1, "", 0, "Show individual sector", min(-1), max(35))
AddOption(drawRelatedSectors, int32_t, 0, "", 0, "Show related sectors (if drawSector != -1)")
AddOption(drawGrid, int32_t, 0, "", 0, "Highlight grid")
AddOption(propagateTracks, int32_t, 0, "", 0, "Propagate final tracks further (inward / outward / show MC tracks)", min(0), max(3))
AddOption(showCollision, int32_t, -1, "", 0, "Show only individual collision")
AddOption(colorCollisions, int32_t, 0, "", 0, "Distinguish collisions in timeframe by color")
AddOption(colorClusters, int32_t, 1, "", 0, "Color clusters belonging to track objects")
AddOption(pointSize, float, 2.0f, "", 0, "Set point size", min(0.8f), max(20.f))
AddOption(lineWidth, float, 1.4f, "", 0, "Set line width", min(0.4f), max(20.f))
AddOption(drawTPC, bool, true, "", 0, "Enable drawing TPC data")
AddOption(drawTRD, bool, true, "", 0, "Enable drawing TRD data")
AddOption(drawTOF, bool, true, "", 0, "Enable drawing TOF data")
AddOption(drawITS, bool, true, "", 0, "Enable drawing ITS data")
AddOption(drawField, bool, false, "", 0, "Enable drawing magnetic field")
AddOption(bFieldStepSize, float, 5.0f, "", 0, "Set field line step size")
AddOption(bFieldStepCount, int32_t, 100, "", 0, "Set field line step count")
AddOption(bFieldLinesCount, int32_t, 2000, "", 0, "Set field lines count")
AddOption(invertColors, bool, false, "", 0, "Invert colors")
AddHelp("help", 'h')
EndConfig()

// Heavy settings concerning the event display (can be changed only with rebuilding vertices)
BeginSubConfig(GPUSettingsDisplayHeavy, heavy, configStandalone.display, "GLH", 0, "Heavy OpenGL display settings", display_heavy)
AddOption(drawTPCTracks, bool, true, "", 0, "Show tracks with TPC contribution")
AddOption(drawITSTracks, bool, true, "", 0, "Show tracks with ITS contribution")
AddOption(drawTRDTracks, bool, true, "", 0, "Show tracks with TRD contribution")
AddOption(drawTOFTracks, bool, true, "", 0, "Show tracks with TOF contribution")
AddOption(drawTracksAndFilter, bool, false, "", 0, "Use AND filter instead of OR filter for selecting tracks")
AddOption(propagateLoopers, bool, false, "", 0, "Enable propagation of loopers")
AddOption(clustersOnly, bool, false, "", 0, "Visualize clusters only")
AddOption(clustersOnNominalRow, bool, false, "", 0, "Show clusters at nominal x of pad row for early-transformed data")
AddOption(separateExtrapolatedTracks, bool, false, "", 0, "Draw track segments propagated to adjacent sectors separately")
AddOption(splitCETracks, int8_t, -1, "", 0, "Split CE tracks when they cross the central electrode (-1 = for triggered data)")
AddOption(markClusters, int32_t, 0, "", 0, "Mark clusters")
AddOption(markFakeClusters, int32_t, 0, "", 0, "Mark fake clusters")
AddOption(markAdjacentClusters, int32_t, 0, "", 0, "Mark adjacent clusters")
AddOption(hideRejectedClusters, int32_t, 1, "", 0, "Hide rejected clusters")
AddOption(hideRejectedTracks, int32_t, 1, "", 0, "Hide rejected tracks")
AddOption(hideUnmatchedClusters, int32_t, 0, "", 0, "Hide unmatched clusters")
AddOption(trackFilter, int32_t, 0, "", 0, "Apply filter on tracks to be displayed")
AddOption(projectXY, int32_t, 0, "", 0, "Project everything on the XY-plane")
AddOption(xzAdd, float, 0, "", 0, "Separate sectors, increase X coordinate", min(0.f), max(300.f))
AddHelp("help", 'h')
EndConfig()

// Camera, window, and renderer settings for the event display
BeginSubConfig(GPUSettingsDisplayRenderer, renderer, configStandalone.display, "GLR", 'g', "Camera / window / renderer OpenGL display settings", display_camera)
AddOption(camLookOrigin, bool, false, "", 0, "Make the camera look at the origin")
AddOption(camYUp, bool, false, "", 0, "Orient the camera such that the y-axis is always upwards")
AddOption(cameraMode, int32_t, 0, "", 0, "Camera mode", min(0), max(3))
AddOption(fullScreen, bool, false, "", 0, "Full Screen")
AddOption(maximized, bool, true, "", 0, "Maximized window")
AddOption(drawQualityMSAA, int32_t, 0, "", 0, "MultiSample Anti Aliasing", min(0), max(16))
AddOption(drawQualityDownsampleFSAA, int32_t, 0, "", 0, "Downsampling Anti Aliasing", min(0), max(4))
AddOption(drawQualityVSync, bool, true, "", 0, "Enable Vertical Sync")
AddOption(fpsLimit, int32_t, 1, "", 0, "FPS limit: 0 = none (maximum rate), 1 = display refresh rate (no limit without VSync), 2 = fpsRate")
AddOption(fpsRate, int32_t, 100, "", 0, "Target frame rate (frames per second) for fpsLimit 2, or if the backend cannot limit to the refresh rate", min(1), max(9999))
AddOption(useGLIndirectDraw, bool, true, "", 0, "Use OpenGL indirect draws to reduce number of draw calls")
AddOption(screenshotScaleFactor, int32_t, 1, "", 0, "Resolution scale factor when taking screenshots", min(1), max(4))
AddOption(fov, float, 45.f, "", 0, "Display FOV", min(5.f), max(175.f))
AddHelp("help", 'h')
EndConfig()

// Vulkan Display Settings
BeginSubConfig(GPUSettingsDisplayVulkan, vulkan, configStandalone.display, "GLV", 0, "Vulkan display settings", display_vulkan)
AddOption(nFramesInFlight, uint32_t, 3, "", 0, "Max number of render frames in flight")
AddOption(uniformBuffersInDeviceMemory, bool, 1, "", 0, "Have uniform buffers in host-accessible device memory")
AddOption(forceDevice, int32_t, -1, "", 0, "Force Vulkan device number to use")
AddHelp("help", 'h')
EndConfig()

// Internal display settings (GUI protocol, console, info texts)
BeginSubConfig(GPUSettingsDisplayInternal, internal, configStandalone.display, "GLI", 0, "Internal display settings", display_internal)
AddOption(guiReplyTimeout, uint32_t, 100, "", 0, "Time in ms the Qt GUI waits at most for the display to process its update")
AddOption(guiMaxUpdatesAhead, uint32_t, 3, "", 0, "Number of updates the Qt GUI may send ahead of the last one the display has processed, before it waits for the display")
AddOption(consoleDelay, float, 0.3f, "", 0, "Time in seconds to open / close the console (^ key)")
AddOption(consoleAlpha, float, 0.85f, "", 0, "Opacity of the console background")
AddOption(consoleVertical, float, 0.5f, "", 0, "Fraction of the window height covered by the fully opened console")
AddOption(consoleHistorySize, uint32_t, 1000000, "", 0, "Number of console lines kept, the history is shortened to this when it reaches twice the size")
AddOption(setInfoTime, float, 5.0f, "", 0, "Time in seconds an info message (SetInfo) is shown fully opaque")
AddOption(setInfoTransition, float, 1.0f, "", 0, "Time in seconds an info message fades out after setInfoTime")
AddOption(consolePrintToTerminal, bool, false, "", 0, "Also write the replies of console commands to the terminal")
AddHelp("help", 'h')
EndConfig()

// Settings concerning the event display (fixed settings, cannot be changed)
BeginSubConfig(GPUSettingsDisplay, display, configStandalone, "GL", 'g', "OpenGL display settings", display)
AddOption(showTPCTracksFromO2Format, bool, false, "", 0, "Use TPC tracks in O2 output format instead of GPU format")
AddOption(font, std::string, "monospace", "", 0, "Font (search patter used for Fontconfig)")
AddOption(fontSize, int32_t, 14, "", 0, "Font size")
AddOption(smoothFont, int32_t, -1, "", 0, "Smoth font when rendering (-1 for auto-select based on size")
AddOption(noFreetype, bool, false, "", 0, "Do not use Freetype for font rendering (can only draw text if supported by frontend)")
AddOption(displayRenderer, std::string, "auto", "renderer", 0, "Renderer for event display: opengl | vulkan | auto")
AddOption(displayFrontend, std::string, "auto", "displayFrontend", 0, "Frontend to use for event display: auto | win32 | x11 | glut | glfw | wayland")
AddOptionVec(filterMacros, std::string, "", 0, "ROOT macros used as track filter")
AddOption(listFrontendOptionHandling, bool, false, "", 0, "List all assigned keys in frontend, Configurable settings in Qt GUI, and missing configuration options")
AddOption(runScript, std::string, "", "", 0, "Console script file, run before the first frame of the display")
AddSubConfig(GPUSettingsDisplayLight, light)
AddSubConfig(GPUSettingsDisplayHeavy, heavy)
AddSubConfig(GPUSettingsDisplayRenderer, renderer)
AddSubConfig(GPUSettingsDisplayVulkan, vulkan)
AddSubConfig(GPUSettingsDisplayInternal, internal)
AddHelp("help", 'h')
EndConfig()

EndNamespace() // o2::gpu

#endif // (!defined(__CLING__) && !defined(G__DICTIONARY)) || defined(BeginSubConfig)
