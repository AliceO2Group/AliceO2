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

/// \file Screenshot.cxx
/// \brief Screenshot functionality
/// \author julian.myrcha@cern.ch
/// \author m.chwesiuk@cern.ch

#include <EventVisualisationView/Screenshot.h>
#include <EventVisualisationView/MultiView.h>
#include "EventVisualisationView/Options.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include <Rtypes.h>
#include "TROOT.h"
#include <filesystem>
#include <sstream>
#include <fairlogger/Logger.h>

namespace o2
{
namespace event_visualisation
{

bool Screenshot::CopyImage(TASImage* dst, TASImage* src, Int_t x_dst, Int_t y_dst, Int_t x_src, Int_t y_src,
                           UInt_t w_src, UInt_t h_src)
{
  if (!dst) {
    return false;
  }
  if (!src) {
    return false;
  }

  int x = 0;
  int y = 0;
  int idx_src = 0;
  int idx_dst = 0;
  x_src = x_src < 0 ? 0 : x_src;
  y_src = y_src < 0 ? 0 : y_src;

  if ((x_src >= (int)src->GetWidth()) || (y_src >= (int)src->GetHeight())) {
    return false;
  }

  w_src = x_src + w_src > src->GetWidth() ? src->GetWidth() - x_src : w_src;
  h_src = y_src + h_src > src->GetHeight() ? src->GetHeight() - y_src : h_src;
  UInt_t yy = (y_src + y) * src->GetWidth();

  src->BeginPaint(false);
  dst->BeginPaint(false);

  UInt_t* dst_image_array = dst->GetArgbArray();
  UInt_t* src_image_array = src->GetArgbArray();

  if (!dst_image_array || !src_image_array) {
    return false;
  }

  for (y = 0; y < (int)h_src; y++) {
    for (x = 0; x < (int)w_src; x++) {

      idx_src = yy + x + x_src;
      idx_dst = (y_dst + y) * dst->GetWidth() + x + x_dst;

      if ((x + x_dst < 0) || (y_dst + y < 0) ||
          (x + x_dst >= (int)dst->GetWidth()) || (y + y_dst >= (int)dst->GetHeight())) {
        continue;
      }

      dst_image_array[idx_dst] = src_image_array[idx_src];
    }
    yy += src->GetWidth();
  }

  return true;
}

TASImage*
  Screenshot::ScaleImage(TASImage* image, UInt_t desiredWidth, UInt_t desiredHeight, const std::string& backgroundColor)
{
  if (!image) {
    return nullptr;
  }
  if (desiredWidth == 0 || desiredHeight == 0) {
    return nullptr;
  }

  UInt_t scaleWidth = desiredWidth;
  UInt_t scaleHeight = desiredHeight;
  UInt_t offsetWidth = 0;
  UInt_t offsetHeight = 0;

  float aspectRatio = (float)image->GetWidth() / (float)image->GetHeight();

  if (desiredWidth >= aspectRatio * desiredHeight) {
    scaleWidth = (UInt_t)(aspectRatio * desiredHeight);
    offsetWidth = (desiredWidth - scaleWidth) / 2.0f;
  } else {
    scaleHeight = (UInt_t)((1.0f / aspectRatio) * desiredWidth);
    offsetHeight = (desiredHeight - scaleHeight) / 2.0f;
  }

  TASImage* scaledImage = new TASImage(desiredWidth, desiredHeight);
  scaledImage->FillRectangle(backgroundColor.c_str(), 0, 0, desiredWidth, desiredHeight);

  image->Scale(scaleWidth, scaleHeight);

  CopyImage(scaledImage, image, offsetWidth, offsetHeight, 0, 0, scaleWidth, scaleHeight);

  return scaledImage;
}

std::string
  Screenshot::perform(const char* prefix, std::string fileName, o2::detectors::DetID::mask_t detectorsMask, int runNumber,
                      int firstTFOrbit, const std::string& collisionTime)
{
  auto start = std::chrono::high_resolution_clock::now();
  UInt_t width = ConfigurationManager::getScreenshotWidth(prefix);
  UInt_t height = ConfigurationManager::getScreenshotHeight(prefix);
  int backgroundColor = ConfigurationManager::getBackgroundColor(); // black
  std::string backgroundColorHex = "#000000";                     // "#19324b";
  TColor* col = gROOT->GetColor(backgroundColor);
  if (col) {
    backgroundColorHex = col->AsHexString();
  }

  auto image = new TASImage(width, height);

  image->FillRectangle(backgroundColorHex.c_str(), 0, 0, width, height);

  auto pixel_object_scale_3d = ConfigurationManager::getScreenshotPixelObjectScale3d(prefix);
  auto pixel_object_scale_rphi = ConfigurationManager::getScreenshotPixelObjectScaleRphi(prefix);
  auto pixel_object_scale_zy = ConfigurationManager::getScreenshotPixelObjectScaleZY(prefix);

  const auto annotationStateTop = MultiView::getInstance()->getAnnotationTop()->GetState();
  const auto annotationStateBottom = MultiView::getInstance()->getAnnotationBottom()->GetState();
  MultiView::getInstance()->getAnnotationTop()->SetState(TGLOverlayElement::kInvisible);
  MultiView::getInstance()->getAnnotationBottom()->SetState(TGLOverlayElement::kInvisible);
  TImage* view3dImage = MultiView::getInstance()->getView(MultiView::EViews::View3d)->GetGLViewer()->GetPictureUsingFBO(width * 0.65, height * 0.95, pixel_object_scale_3d);
  MultiView::getInstance()->getAnnotationTop()->SetState(annotationStateTop);
  MultiView::getInstance()->getAnnotationBottom()->SetState(annotationStateBottom);
  CopyImage(image, (TASImage*)view3dImage, width * 0.015, height * 0.025, 0, 0, view3dImage->GetWidth(),
            view3dImage->GetHeight());
  delete view3dImage;

  TImage* viewRphiImage = MultiView::getInstance()->getView(
                                                    MultiView::EViews::ViewRphi)
                            ->GetGLViewer()
                            ->GetPictureUsingFBO(width * 0.3, height * 0.45,
                                                 pixel_object_scale_rphi);
  CopyImage(image, (TASImage*)viewRphiImage, width * 0.68, height * 0.025, 0, 0, viewRphiImage->GetWidth(),
            viewRphiImage->GetHeight());
  delete viewRphiImage;

  TImage* viewZYImage = MultiView::getInstance()->getView(MultiView::EViews::ViewZY)->GetGLViewer()->GetPictureUsingFBO(width * 0.3, height * 0.45, pixel_object_scale_zy);
  CopyImage(image, (TASImage*)viewZYImage, width * 0.68, height * 0.525, 0, 0, viewZYImage->GetWidth(),
            viewZYImage->GetHeight());
  delete viewZYImage;

  static std::map<std::string, TASImage*> aliceLogos;
  TASImage* aliceLogo = nullptr;
  if (aliceLogos.find(prefix) != aliceLogos.end()) {
    aliceLogo = aliceLogos.find(prefix)->second;
  }

  if (aliceLogo == nullptr) {
    aliceLogo = new TASImage(ConfigurationManager::getScreenshotLogoAlice());
    if (aliceLogo->IsValid()) {
      double ratio = (double)(aliceLogo->GetWidth()) / (double)(aliceLogo->GetHeight());
      aliceLogo->Scale(0.08 * width, 0.08 * width / ratio);
      aliceLogos[prefix] = aliceLogo;
    }
  }
  if (aliceLogo != nullptr) {
    if (aliceLogo->IsValid()) {
      image->Merge(aliceLogo, "alphablend", 0.01 * width, 0.01 * width);
    }
  }
  // delete aliceLogo; /// avoiding reload the same resource

  int o2LogoMarginX = 0.01 * width;
  int o2LogoMarginY = 0.01 * width;
  int o2LogoSize = 0.04 * width;

  static std::map<std::string, TASImage*> o2Logos;
  TASImage* o2Logo = nullptr;
  if (o2Logos.find(prefix) != o2Logos.end()) {
    o2Logo = o2Logos.find(prefix)->second;
  }

  if (o2Logo == nullptr) {
    o2Logo = new TASImage(ConfigurationManager::getScreenshotLogoO2());
    if (o2Logo->IsValid()) {
      double ratio = (double)(o2Logo->GetWidth()) / (double)(o2Logo->GetHeight());
      o2Logo->Scale(o2LogoSize, o2LogoSize / ratio);
      o2Logos[prefix] = o2Logo;
    }
  }

  if (o2Logo != nullptr) {
    if (o2Logo->IsValid()) {
      double ratio = (double)(o2Logo->GetWidth()) / (double)(o2Logo->GetHeight());
      image->Merge(o2Logo, "alphablend", o2LogoMarginX, height - o2LogoSize / ratio - o2LogoMarginY);
    }
  }

  int textX = o2LogoMarginX + o2LogoSize + o2LogoMarginX;
  int textY = height - o2LogoSize - o2LogoMarginY;

  auto detectorsString = detectors::DetID::getNames(detectorsMask);

  std::vector<std::string> lines;
  if (!collisionTime.empty()) {
    lines.push_back(
      (std::string)ConfigurationManager::getInstance().getSettings().GetValue("screenshot.message.line.0",
                                                                              TString::Format("Run number: %d",
                                                                                              runNumber)));
    lines.push_back(
      (std::string)ConfigurationManager::getInstance().getSettings().GetValue("screenshot.message.line.1",
                                                                              TString::Format("First TF orbit: %d",
                                                                                              firstTFOrbit)));
    lines.push_back(
      (std::string)ConfigurationManager::getInstance().getSettings().GetValue("screenshot.message.line.2",
                                                                              TString::Format("Date: %s",
                                                                                              collisionTime.c_str())));
    lines.push_back(
      (std::string)ConfigurationManager::getInstance().getSettings().GetValue("screenshot.message.line.3",
                                                                              TString::Format("Detectors: %s",
                                                                                              detectorsString.c_str())));
  }

  image->BeginPaint();
  int fontSize = 0.015 * height;
  int textLineHeight = 0.015 * height;

  for (int i = 0; i < 4; i++) {
    lines.push_back(""); // guard for empty collision time
    image->DrawText(textX, textY + i * textLineHeight, lines[i].c_str(), fontSize, "#BBBBBB", "FreeSansBold.otf");
  }
  image->EndPaint();

  image->WriteImage(fileName.c_str(), TImage::kPng);
  delete image;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  return fileName; // saved screenshod file name
}

} // namespace event_visualisation
} // namespace o2
