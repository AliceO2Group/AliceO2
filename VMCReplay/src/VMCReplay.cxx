// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "VMCReplay/VMCReplay.h"

VMCReplay::VMC::Replay(const std::string& inputTreeName)
  : TGeant3TGeo("VMCReplay"), mChain(inputTreeName.c_str())
{
}

VMCReplay::AddInputFiles(std::initializer_list<std::string> inputFileNames)
{
  for (auto& f : inputFileNames) {
    mChain.Add(f.c_str());
  }
}

ClassImp(VMCReplay);
