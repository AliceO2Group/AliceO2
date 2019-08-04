// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    ConfigurationManager.h
/// \author  Jeremi Niedziela
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_CONFIGURATIONMANAGER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_CONFIGURATIONMANAGER_H

#include <TEnv.h>

namespace o2
{
namespace event_visualisation
{

/// Configuration Manager allows an easy access to the config file.
///
/// Configuration Manager is a singleton which assures an access to
/// the correct configuration file, regardless wether it is located
/// in the users home directory or in the O2 installation path.

class ConfigurationManager
{
 public:
  /// Returns an instance of ConfigurationManager
  static ConfigurationManager& getInstance();

  /// Returns current event display configuration
  void getConfig(TEnv& settings) const;

 private:
  /// Default constructor
  ConfigurationManager() = default;
  /// Default destructor
  ~ConfigurationManager() = default;
  /// Deleted copy constructor
  ConfigurationManager(ConfigurationManager const&) = delete;
  /// Deleted assignment operator
  void operator=(ConfigurationManager const&) = delete;
};

#endif
}
}
