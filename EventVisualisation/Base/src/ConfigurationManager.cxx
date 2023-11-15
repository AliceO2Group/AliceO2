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

///
/// \file    ConfigurationManager.cxx
/// \author  Jeremi Niedziela
///

#include "EventVisualisationBase/ConfigurationManager.h"
#include <fairlogger/Logger.h>
#include <TSystem.h>

using namespace std;

namespace o2
{
namespace event_visualisation
{

ConfigurationManager& ConfigurationManager::getInstance()
{
  static ConfigurationManager instance;
  return instance;
}

void ConfigurationManager::getConfig(TEnv& settings) const
{
  TString fileName;
  if (not this->mOptionsFileName.empty() and settings.ReadFile(fileName = this->mOptionsFileName, kEnvUser) >= 0) {
    return; // precise located options file name read succesfully
  }
  if (settings.ReadFile(fileName = ".o2eve_config", kEnvUser) < 0) {
    LOGF(warn, "could not find .o2eve_config in working directory! Trying .o2eve_config in home directory");
    if (settings.ReadFile(fileName = Form("%s/.o2eve_config", gSystem->Getenv("HOME")), kEnvUser) < 0) {
      LOGF(warn, "could not find .o2eve_config in home directory! Trying o2eve_config in home directory");
      if (settings.ReadFile(fileName = Form("%s/o2eve_config", gSystem->Getenv("HOME")), kEnvUser) < 0) {
        LOGF(warn, "could not find o2eve_config in home directory! Trying o2eve_config in O2/EventVisualisation");
        if (settings.ReadFile(fileName = Form("%s/EventVisualisation/o2eve_config",
                                              gSystem->Getenv("ALICEO2_INSTALL_PATH")),
                              kEnvUser) < 0) {
          LOGF(fatal, "could not find .o2eve_config or o2eve_config file!.");
          exit(0);
        }
      }
    }
  }
  LOGF(info, Form("using %s config settings", fileName.Data()));
}

const ConfigurationManager* ConfigurationManager::loadSettings()
{
  if (this->mSettingsLoadCounter <= 0) {
    this->getConfig(mSettings);
    this->mSettingsLoadCounter = mSettings.GetValue("settings.reload", 10000);
  }
  this->mSettingsLoadCounter--;
  return this;
}

UInt_t ConfigurationManager::getRefreshRateInSeconds()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("refresh.rate", 5);
}

UInt_t ConfigurationManager::getOutreachFrequencyInRefreshRates() // 1 means skip one refresh
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("outreach.frequency.refresh.rate", 1);
}

std::string ConfigurationManager::getScreenshotPath(const char* prefix)
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue((std::string(prefix) + ".path").c_str(), "Screenshots");
}

UInt_t ConfigurationManager::getOutreachFilesMax()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("outreach.files.max", 10);
}

UInt_t ConfigurationManager::getScreenshotWidth(const char* prefix)
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue((std::string(prefix) + ".width").c_str(), 3840);
}

UInt_t ConfigurationManager::getScreenshotHeight(const char* prefix)
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue((std::string(prefix) + ".height").c_str(), 2160);
}

UInt_t ConfigurationManager::getScreenshotPixelObjectScale3d(const char* prefix)
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue((std::string(prefix) + ".pixel_object_scale.3d").c_str(), 0);
}

UInt_t ConfigurationManager::getScreenshotPixelObjectScaleRphi(const char* prefix)
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue((std::string(prefix) + ".pixel_object_scale.rphi").c_str(), 0);
}

UInt_t ConfigurationManager::getScreenshotPixelObjectScaleZY(const char* prefix)
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue((std::string(prefix) + ".pixel_object_scale.zy").c_str(), 0);
}

bool ConfigurationManager::getScreenshotMonthly()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("screenshot.monthly", 0);
}

std::string ConfigurationManager::getDataDefault()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("data.default", "NEWEST");
}

std::string ConfigurationManager::getDataSyntheticRunDir()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("data.synthetic.run.dir",
                                                                                "jsons/synthetic");
}

std::string ConfigurationManager::getDataCosmicRunDir()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("data.cosmics.run.dir",
                                                                                "jsons/cosmics");
}

std::string ConfigurationManager::getDataPhysicsRunDir()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("data.physics.run.dir",
                                                                                "jsons/physics");
}

std::string ConfigurationManager::getSimpleGeomR3Path()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("simple.geom.R3.path", "");
}

UInt_t ConfigurationManager::getBackgroundColor()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("background.color", 1);
}

bool ConfigurationManager::getAxesShow()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("simple.geom.R3.path", "");
}

bool ConfigurationManager::getFullScreenMode()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("simple.geom.R3.path", "");
}

double ConfigurationManager::getCamera3DRotationHorizontal()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("camera.3D.rotation.horizontal", -0.4);
}

double ConfigurationManager::getCamera3DRotationVertical()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("camera.3D.rotation.vertical", 1.0);
}

double ConfigurationManager::getCamera3DZoom()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("camera.3D.zoom", 1.0);
}

double ConfigurationManager::getCameraRPhiZoom()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("camera.R-Phi.zoom", 1.0);
}

double ConfigurationManager::getCameraZYZoom()
{
  return ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("camera.Z-Y.zoom", 1.0);
}

const char* ConfigurationManager::getScreenshotLogoO2()
{
  static std::string str = ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue("screenshot.logo.o2",
                                                                                                  "o2.png");
  return str.c_str();
}

const char* ConfigurationManager::getScreenshotLogoAlice()
{
  static std::string str = ConfigurationManager::getInstance().loadSettings()->mSettings.GetValue(
    "screenshot.logo.alice", "alice-white.png");
  return str.c_str();
}

} // namespace event_visualisation
} // namespace o2
