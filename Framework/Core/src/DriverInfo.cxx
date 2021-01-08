// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DriverInfo.h"

char const* o2::framework::DriverInfoHelper::stateToString(enum DriverState state)
{
  static const char* names[static_cast<int>(DriverState::LAST)] = {
    "INIT",                    //
    "SCHEDULE",                //
    "RUNNING",                 //
    "REDEPLOY_GUI",            //
    "QUIT_REQUESTED",          //
    "HANDLE_CHILDREN",         //
    "EXIT",                    //
    "UNKNOWN",                 //
    "PERFORM_CALLBACKS",       //
    "MATERIALISE_WORKFLOW",    //
    "IMPORT_CURRENT_WORKFLOW", //
    "DO_CHILD"                 //
  };
  return names[static_cast<int>(state)];
}
