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
/// \file    ConfigurationManager.h
/// \author  Jeremi Niedziela
/// \author  Julian Myrcha
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_CONFIGURATIONMANAGER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_CONFIGURATIONMANAGER_H

#include <TEnv.h>

namespace o2
{
namespace event_visualisation
{
/// Version of the software
const static int o2_eve_version = 172;

/// Configuration Manager allows an easy access to the config file.
///
/// Configuration Manager is a singleton which assures an access to
/// the correct configuration file, regardless of where it is located
/// in the users home directory or in the O2 installation path.

class ConfigurationManager
{
 private:
  std::string mOptionsFileName;
  TEnv mSettings;
  int mSettingsLoadCounter; /// used to reduce loading
  const ConfigurationManager* loadSettings();

 public:
  /// Returns an instance of ConfigurationManager
  static ConfigurationManager& getInstance();

  const TEnv& getSettings()
  {
    return loadSettings()->mSettings;
  }

  /// sets precise location of option file name (if not empty only this is used)
  static void setOptionsFileName(const std::string& fileName)
  {
    getInstance().mOptionsFileName = fileName;
  }

  /// Returns current event display configuration
  void getConfig(TEnv& settings) const;

  static UInt_t getRefreshRateInSeconds();

  static std::string getScreenshotPath(const char* prefix);
  static UInt_t getScreenshotWidth(const char* prefix);
  static UInt_t getScreenshotHeight(const char* prefix);

  static UInt_t getScreenshotPixelObjectScale3d(const char* prefix);
  static UInt_t getScreenshotPixelObjectScaleRphi(const char* prefix);
  static UInt_t getScreenshotPixelObjectScaleZY(const char* prefix);

  static std::string getDataDefault();
  static std::string getDataSyntheticRunDir();
  static std::string getDataCosmicRunDir();
  static std::string getDataPhysicsRunDir();
  static std::string getSimpleGeomR3Path();
  static UInt_t getOutreachFilesMax();

  static UInt_t getOutreachFrequencyInRefreshRates();
  static bool getScreenshotMonthly();
  static UInt_t getBackgroundColor();
  static bool getAxesShow();
  static bool getFullScreenMode();
  static double getCamera3DRotationHorizontal();
  static double getCamera3DRotationVertical();
  static double getCamera3DZoom();
  static double getCameraRPhiZoom();
  static double getCameraZYZoom();
  static const char* getScreenshotLogoO2();
  static const char* getScreenshotLogoAlice();

  /// Deleted copy constructor
  ConfigurationManager(ConfigurationManager const&) = delete;
  /// Deleted assignment operator
  void operator=(ConfigurationManager const&) = delete;

 private:
  /// Default constructor
  ConfigurationManager() = default;
  /// Default destructor
  ~ConfigurationManager() = default;
};

}
}
#endif // ALICE_O2_EVENTVISUALISATION_BASE_CONFIGURATIONMANAGER_H
