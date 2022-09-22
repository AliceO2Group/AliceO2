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
#include <fstream>
#include <sstream>

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

TASImage* Screenshot::ScaleImage(TASImage* image, UInt_t desiredWidth, UInt_t desiredHeight, const std::string& backgroundColor)
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

std::string Screenshot::perform(std::string fileName, o2::detectors::DetID::mask_t detectorsMask, int runNumber, int firstTFOrbit, std::string collisionTime)
{
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  UInt_t width = settings.GetValue("screenshot.width", 3840);
  UInt_t height = settings.GetValue("screenshot.height", 2160);
  int backgroundColor = settings.GetValue("background.color", 1); // black
  std::string backgroundColorHex = "#000000";                     // "#19324b";
  TColor* col = gROOT->GetColor(backgroundColor);
  if (col) {
    backgroundColorHex = col->AsHexString();
  }

  std::string outDirectory = settings.GetValue("screenshot.path", "Screenshots");

  std::time_t time = std::time(nullptr);
  char time_str[100];
  std::strftime(time_str, sizeof(time_str), "%Y_%m_%d_%H_%M_%S", std::localtime(&time));

  TASImage* scaledImage;

  bool monthDirectory = settings.GetValue("screenshot.monthly", 0);

  if (monthDirectory) {
    char dir_str[32];
    std::strftime(dir_str, sizeof(dir_str), "%Y-%d", std::localtime(&time));
    outDirectory = outDirectory + "/" + dir_str;
    std::filesystem::create_directory(outDirectory);
  }

  std::ostringstream filepath;
  filepath << outDirectory << "/Screenshot_" << time_str << ".png";

  TASImage* image = new TASImage(width, height);

  image->FillRectangle(backgroundColorHex.c_str(), 0, 0, width, height);

  const auto annotationStateTop = MultiView::getInstance()->getAnnotationTop()->GetState();
  const auto annotationStateBottom = MultiView::getInstance()->getAnnotationBottom()->GetState();
  MultiView::getInstance()->getAnnotationTop()->SetState(TGLOverlayElement::kInvisible);
  MultiView::getInstance()->getAnnotationBottom()->SetState(TGLOverlayElement::kInvisible);
  TImage* view3dImage = MultiView::getInstance()->getView(MultiView::EViews::View3d)->GetGLViewer()->GetPictureUsingFBO(width * 0.65, height * 0.95);
  MultiView::getInstance()->getAnnotationTop()->SetState(annotationStateTop);
  MultiView::getInstance()->getAnnotationBottom()->SetState(annotationStateBottom);
  CopyImage(image, (TASImage*)view3dImage, width * 0.015, height * 0.025, 0, 0, view3dImage->GetWidth(), view3dImage->GetHeight());
  delete view3dImage;

  TImage* viewRphiImage = MultiView::getInstance()->getView(MultiView::EViews::ViewRphi)->GetGLViewer()->GetPictureUsingFBO(width * 0.3, height * 0.45);
  CopyImage(image, (TASImage*)viewRphiImage, width * 0.68, height * 0.025, 0, 0, viewRphiImage->GetWidth(), viewRphiImage->GetHeight());
  delete viewRphiImage;

  TImage* viewZYImage = MultiView::getInstance()->getView(MultiView::EViews::ViewZY)->GetGLViewer()->GetPictureUsingFBO(width * 0.3, height * 0.45);
  CopyImage(image, (TASImage*)viewZYImage, width * 0.68, height * 0.525, 0, 0, viewZYImage->GetWidth(), viewZYImage->GetHeight());
  delete viewZYImage;

  TASImage* aliceLogo = new TASImage(settings.GetValue("screenshot.logo.alice", "alice-white.png"));
  if (aliceLogo->IsValid()) {
    double ratio = (double)(aliceLogo->GetWidth()) / (double)(aliceLogo->GetHeight());
    aliceLogo->Scale(0.08 * width, 0.08 * width / ratio);
    image->Merge(aliceLogo, "alphablend", 0.01 * width, 0.01 * width);
  }
  delete aliceLogo;

  TASImage* o2Logo = new TASImage(settings.GetValue("screenshot.logo.o2", "o2.png"));
  int o2LogoMarginX = 0.01 * width;
  int o2LogoMarginY = 0.01 * width;
  int o2LogoSize = 0.04 * width;

  if (o2Logo->IsValid()) {
    double ratio = (double)(o2Logo->GetWidth()) / (double)(o2Logo->GetHeight());
    o2Logo->Scale(o2LogoSize, o2LogoSize / ratio);
    image->Merge(o2Logo, "alphablend", o2LogoMarginX, height - o2LogoSize / ratio - o2LogoMarginY);
  }
  delete o2Logo;
  int textX = o2LogoMarginX + o2LogoSize + o2LogoMarginX;
  int textY = height - o2LogoSize - o2LogoMarginY;

  auto detectorsString = detectors::DetID::getNames(detectorsMask);

  std::vector<std::string> lines;
  if (!collisionTime.empty()) {
    lines.push_back((std::string)settings.GetValue("screenshot.message.line.0", TString::Format("Run number: %d", runNumber)));
    lines.push_back((std::string)settings.GetValue("screenshot.message.line.1", TString::Format("First TF orbit: %d", firstTFOrbit)));
    lines.push_back((std::string)settings.GetValue("screenshot.message.line.2", TString::Format("Date: %s", collisionTime.c_str())));
    lines.push_back((std::string)settings.GetValue("screenshot.message.line.3", TString::Format("Detectors: %s", detectorsString.c_str())));
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
  return fileName; // saved screenshod file name
}

} // namespace event_visualisation
} // namespace o2
