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
#include "GPUDisplayMagneticField.h"

#include "GPUDisplayBackendOpenGL.h"

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
#include <string>

using namespace GPUCA_NAMESPACE::gpu;

GPUDisplayBackend::GPUDisplayBackend() = default;
GPUDisplayBackend::~GPUDisplayBackend() = default;

GPUDisplayBackend* GPUDisplayBackend::getBackend(const char* type)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
  if (strcmp(type, "vulkan") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayBackendVulkan;
  } else
#endif
  if (strcmp(type, "opengl") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayBackendOpenGL;
  } else {
    GPUError("Requested renderer not available");
  }
  return nullptr;
}

int GPUDisplayBackend::InitBackend()
{
  int retVal = InitBackendA();
  if (retVal) {
    return retVal;
  }
  if (mDisplay->cfg().noFreetype) {
    return retVal;
  }
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
    return 0;
  }
  if (FT_New_Face(ft, fontName.c_str(), 0, &face)) {
    GPUError("Error loading freetypoe font");
    return 0;
  }

  int fontSize = mDisplay->cfg().fontSize;
  mDisplay->drawTextFontSize() = fontSize;
  if (smoothFont()) {
    fontSize *= 4; // Font size scaled by 4, can be downsampled
  }
  FT_Set_Pixel_Sizes(face, 0, fontSize);

  for (unsigned int i = 0; i < 128; i++) {
    if (FT_Load_Char(face, i, FT_LOAD_RENDER)) {
      GPUError("Error loading freetype symbol");
      return 0;
    }
    const auto& glyph = face->glyph;
    addFontSymbol(i, glyph->bitmap.width, glyph->bitmap.rows, glyph->bitmap_left, glyph->bitmap_top, glyph->advance.x, glyph->bitmap.buffer);
  }
  initializeTextDrawing();
  FT_Done_Face(face);
  FT_Done_FreeType(ft);
  mFreetypeInitialized = true;
#endif // GPUCA_BUILD_EVENT_DISPLAY_FREETYPE
  return retVal;
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
  mCmdBuffer.clear();
  mIndirectSliceOffset.resize(GPUCA_NSLICES);
  // TODO: Check if this can be parallelized
  for (int iSlice = 0; iSlice < GPUCA_NSLICES; iSlice++) {
    mIndirectSliceOffset[iSlice] = mCmdBuffer.size();
    for (unsigned int k = 0; k < mDisplay->vertexBufferStart()[iSlice].size(); k++) {
      mCmdBuffer.emplace_back(mDisplay->vertexBufferCount()[iSlice][k], 1, mDisplay->vertexBufferStart()[iSlice][k], 0);
    }
  }
}

float GPUDisplayBackend::getDownsampleFactor(bool screenshot)
{
  float factor = 1.0f;
  int fsaa = mDisplay->cfgR().drawQualityDownsampleFSAA;
  int screenshotScale = mDisplay->cfgR().screenshotScaleFactor;
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
