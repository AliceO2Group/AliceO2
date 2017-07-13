// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file    EveInitializer.h
/// @author  Jeremi Niedziela
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_EVEINITIALIZER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_EVEINITIALIZER_H

#include <TString.h>

class EveInitializer
{
public:
  EveInitializer(const EveEventManager::EDataSource defaultDataSource= EveEventManager::kSourceOffline);
  ~EveInitializer();
  
//  static void GetConfig(TEnv *settings){}
//  static void SetupGeometry(){}
//  static void AddMacros(){}
//  static void SetupCamera(){}
//  static void SetupBackground(){}
private:
//  void ImportMacros(){}
};

#endif
