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

/// \file GPUDisplayBackend.cxx
/// \author David Rohr

#include "GPUDisplayBackend.h"
#include "helpers/GPUDisplayMagneticField.h"

#include "GPUDisplayBackendNone.h"
#ifdef GPUCA_BUILD_EVENT_DISPLAY_OPENGL
#include "GPUDisplayBackendOpenGL.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
#include "GPUDisplayBackendVulkan.h"
#endif

#ifdef GPUCA_BUILD_EVENT_DISPLAY_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#ifdef GPUCA_BUILD_EVENT_DISPLAY_FONTCONFIG
#if !__has_include(<fontconfig/fontconfig.h>)
#undef GPUCA_BUILD_EVENT_DISPLAY_FONTCONFIG
#else
#include <fontconfig/fontconfig.h>
#endif
#endif
#endif

#include "GPUDisplay.h"
#include "GPULogging.h"
#include "GPUTPCGeometry.h"
#include <string>

using namespace o2::gpu;

GPUDisplayBackend::GPUDisplayBackend() = default;
GPUDisplayBackend::~GPUDisplayBackend() = default;

GPUDisplayBackend* GPUDisplayBackend::getBackend(const char* type)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
  if (strcmp(type, "vulkan") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayBackendVulkan;
  } else
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_OPENGL
    if (strcmp(type, "opengl") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayBackendOpenGL;
  } else
#endif
    if (strcmp(type, "none") == 0) {
    return new GPUDisplayBackendNone;
  } else {
    GPUError("Requested renderer not available");
  }
  return nullptr;
}

int32_t GPUDisplayBackend::InitBackend()
{
  int32_t retVal = InitBackendA();
  if (retVal) {
    return retVal;
  }
  if (!mDisplay->cfg().noFreetype) {
    loadFont();
  }
  return retVal;
}

void GPUDisplayBackend::updateFontScale(float scale)
{
  if (mFreetypeInitialized && scale != mFontRasterScale) {
    clearFontSymbols();
    mFreetypeInitialized = false;
    loadFont();
  }
}

void GPUDisplayBackend::loadFont()
{
  std::string fontName = mDisplay->cfg().font;
#ifdef GPUCA_BUILD_EVENT_DISPLAY_FONTCONFIG
  FcInit();
  FcConfig* config = FcInitLoadConfigAndFonts();
  FcPattern* pat = FcNameParse((const FcChar8*)fontName.c_str());
  FcConfigSubstitute(config, pat, FcMatchPattern);
  FcDefaultSubstitute(pat);
  FcResult result;
  FcPattern* font = FcFontMatch(config, pat, &result);
  if (font && result == 0) {
    FcChar8* file = nullptr;
    if (FcPatternGetString(font, FC_FILE, 0, &file) == FcResultMatch) {
      fontName = (char*)file;
    }
  } else {
    GPUError("Coult not find font for pattern %s", fontName.c_str());
  }
  FcPatternDestroy(font);
  FcPatternDestroy(pat);
  FcConfigDestroy(config);
  FcFini();
#endif // GPUCA_BUILD_EVENT_DISPLAY_FONTCONFIG

#ifdef GPUCA_BUILD_EVENT_DISPLAY_FREETYPE
  FT_Library ft;
  FT_Face face;
  if (FT_Init_FreeType(&ft)) {
    GPUError("Error initializing freetype");
    return;
  }
  if (FT_New_Face(ft, fontName.c_str(), 0, &face)) {
    GPUError("Error loading freetype font");
    FT_Done_FreeType(ft);
    return;
  }

  int32_t fontSize = mDisplay->cfg().fontSize;
  mDisplay->drawTextFontSize() = fontSize;
  if (smoothFont()) {
    fontSize *= 4; // Font size scaled by 4, can be downsampled
  }
  mFontRasterScale = mDisplay->frontend()->getDisplayScale();
  FT_Set_Pixel_Sizes(face, 0, (uint32_t)(fontSize * mFontRasterScale + 0.5f)); // Rasterize at physical pixel size for HiDPI

  for (uint32_t i = 0; i < SOLID_GLYPH; i++) {
    if (FT_Load_Char(face, i, FT_LOAD_RENDER)) {
      GPUError("Error loading freetype symbol");
      FT_Done_Face(face);
      FT_Done_FreeType(ft);
      return;
    }
    const auto& glyph = face->glyph;
    if (i == 'M') {
      mFontCharWidth = (glyph->advance.x >> 6) * (smoothFont() ? 0.25f : 1.f) / mFontRasterScale; // As in GLPrintText of the backends
    }
    addFontSymbol(i, glyph->bitmap.width, glyph->bitmap.rows, glyph->bitmap_left, glyph->bitmap_top, glyph->advance.x, glyph->bitmap.buffer);
  }
  std::vector<uint8_t> solid(8 * 8, 255);
  addFontSymbol(SOLID_GLYPH, 8, 8, 0, 0, 0, solid.data());
  initializeTextDrawing();
  FT_Done_Face(face);
  FT_Done_FreeType(ft);
  mFreetypeInitialized = true;
#endif // GPUCA_BUILD_EVENT_DISPLAY_FREETYPE
}

void GPUDisplayBackend::ExitBackend()
{
  ExitBackendA();
}

std::vector<char> GPUDisplayBackend::getPixels()
{
  auto retVal = std::move(mScreenshotPixels);
  mScreenshotPixels = std::vector<char>();
  return retVal;
}

void GPUDisplayBackend::fillIndirectCmdBuffer()
{
  mIndirectSectorOffset.resize(GPUTPCGeometry::NSECTORS);
  size_t total = 0;
  for (uint32_t iSector = 0; iSector < GPUTPCGeometry::NSECTORS; iSector++) {
    mIndirectSectorOffset[iSector] = total;
    total += mDisplay->vertexBufferStart()[iSector].size();
  }
  mCmdBuffer.resize(total);
  for (uint32_t iSector = 0; iSector < GPUTPCGeometry::NSECTORS; iSector++) {
    for (uint32_t k = 0; k < mDisplay->vertexBufferStart()[iSector].size(); k++) {
      mCmdBuffer[mIndirectSectorOffset[iSector] + k] = DrawArraysIndirectCommand(mDisplay->vertexBufferCount()[iSector][k], 1, mDisplay->vertexBufferStart()[iSector][k], 0);
    }
  }
}

int32_t GPUDisplayBackend::fpsLimitMode() const
{
  const auto& cfg = mDisplay->cfgR();
  if (cfg.fpsLimit == FPS_LIMIT_VSYNC) {
    return !cfg.drawQualityVSync ? FPS_LIMIT_NONE : (canLimitToVSync() ? FPS_LIMIT_VSYNC : FPS_LIMIT_RATE);
  }
  return cfg.fpsLimit;
}

float GPUDisplayBackend::getDownsampleFactor(bool screenshot)
{
  float factor = 1.0f;
  int32_t fsaa = mDisplay->cfgR().drawQualityDownsampleFSAA;
  int32_t screenshotScale = mDisplay->cfgR().screenshotScaleFactor;
  if (fsaa) {
    factor *= fsaa;
  }
  if (screenshotScale && screenshot) {
    factor *= screenshotScale;
  }
  return factor;
}

bool GPUDisplayBackend::smoothFont()
{
  return mDisplay->cfg().smoothFont < 0 ? (mDisplay->cfg().fontSize > 12) : mDisplay->cfg().smoothFont;
}

void GPUDisplayBackend::GLDrawLine(float x0, float y0, float x1, float y1, float width, const float* color)
{
  const float dx = x1 - x0, dy = y1 - y0, length = std::sqrt(dx * dx + dy * dy);
  if (length == 0.f) {
    return;
  }
  const float nx = -dy / length * width * 0.5f, ny = dx / length * width * 0.5f; // Perpendicular, half the width
  GLDrawQuad({{x0 + nx, y0 + ny}, {x0 - nx, y0 - ny}, {x1 - nx, y1 - ny}, {x1 + nx, y1 + ny}}, color);
}

uint32_t GPUDisplayBackend::GLDrawConsole(float verticalFraction, std::span<const std::string> previous, const std::string& current, int32_t cursorPos, uint32_t scroll)
{
  if (!mFreetypeInitialized) {
    return 1;
  }
  const float scale = mDisplay->frontend()->getDisplayScale();
  const float colorValue = mDisplay->cfgL().invertColors ? 0.f : 1.f;
  float textColor[4] = {colorValue, colorValue, colorValue, 1.f};
  const float backgroundColor[4] = {1.f - colorValue, 1.f - colorValue, 1.f - colorValue, mDisplay->cfg().internal.consoleAlpha};
  const float width = mScreenWidth, top = mScreenHeight, bottom = top * (1.f - verticalFraction);
  GLDrawQuad({{0.f, bottom}, {width, bottom}, {width, top}, {0.f, top}}, backgroundColor);
  GLDrawLine(0.f, bottom, width, bottom, scale, textColor);

  static constexpr float MARGIN = 6.f;                                                                                    // Logical pixels from the text to the left border and to the line
  const float lineHeight = std::max(20, mDisplay->drawTextFontSize() + 4) * scale;                                        // As the info text of GPUDisplay
  const float descent = 0.25f * mDisplay->drawTextFontSize() * scale;                                                     // Below the baseline, approximately
  const size_t maxChars = mFontCharWidth > 0.f ? std::max(0.f, (width / scale - 2 * MARGIN) / mFontCharWidth) : SIZE_MAX; // Longer lines are cut
  float y = bottom + (0.5f + MARGIN) * scale + descent;                                                                   // Baseline of the lowest line, MARGIN above the line (half its width)
  if (cursorPos >= 0 && (size_t)cursorPos < maxChars && mFontCharWidth > 0.f) {                                           // Monospace font: position from the character width
    const float x0 = (MARGIN + cursorPos * mFontCharWidth) * scale, x1 = x0 + mFontCharWidth * scale;
    const float y1 = y - 0.5f * descent, y0 = y1 - 3.f * scale;
    GLDrawQuad({{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}, textColor); // Solid underline, 3 logical pixels, below the baseline
  }
  const uint32_t nFit = std::max(1.f, std::floor((top - y + descent) / lineHeight) - 1.f); // Lines above the current one that fit, as in the loop
  scroll = std::min<size_t>(scroll, previous.size());
  for (size_t i = 0; i <= previous.size() - scroll && y + lineHeight - descent <= top; i++, y += lineHeight) {
    const std::string& text = i == 0 ? current : previous[previous.size() - scroll - i];
    if (!text.empty() && maxChars) {
      GLPrintText(text.c_str(), MARGIN * scale, y, textColor, scale / mFontRasterScale, std::min(text.size(), maxChars));
    }
  }
  return nFit;
}
