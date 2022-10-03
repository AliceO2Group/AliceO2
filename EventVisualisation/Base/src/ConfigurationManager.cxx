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
  if (settings.ReadFile(fileName = ".o2eve_config", kEnvUser) < 0) {
    LOG(warn) << "could not find .o2eve_config in working directory! Trying .o2eve_config in home directory";
    if (settings.ReadFile(fileName = Form("%s/.o2eve_config", gSystem->Getenv("HOME")), kEnvUser) < 0) {
      LOG(warn) << "could not find .o2eve_config in home directory! Trying o2eve_config in home directory";
      if (settings.ReadFile(fileName = Form("%s/o2eve_config", gSystem->Getenv("HOME")), kEnvUser) < 0) {
        LOG(warn) << "could not find o2eve_config in home directory! Trying o2eve_config in O2/EventVisualisation";
        if (settings.ReadFile(fileName = Form("%s/EventVisualisation/o2eve_config",
                                              gSystem->Getenv("ALICEO2_INSTALL_PATH")),
                              kEnvUser) < 0) {
          LOG(fatal) << "could not find .o2eve_config or o2eve_config file!.";
          exit(0);
        }
      }
    }
  }
  LOG(info) << Form("using %s config settings", fileName.Data());
}

} // namespace event_visualisation
} // namespace o2
