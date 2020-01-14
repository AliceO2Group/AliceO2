// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_VMCREPLAY_H
#define O2_VMCREPLAY_H

#include <vector>

#include "TChain.h"

#include "MCStepLogger/StepInfo.h"

#include "TGeant3TGeo.h"

class VMCReplay : public TGeant3TGeo
{
 public:
  VMCReplay(const std::string& inputTreeName);

  void AddInputFiles(std::initializer_list<std::string> inputFileNames);

  virtual Bool_t ProcessRun(Int_t nofEvents) override;

 private:
  bool ConnectToStepTree();
  bool ProcessStepTree();

  bool LoadNextStep();
  bool ProcessStep();

 private:
  TChain mChain;

  std::vector<o2::StepInfo>* mStepInfo = nullptr;
  std::vector<o2::MagCallInfo>* mMagInfo = nullptr;
  o2::StepLookups* mLookups = nullptr;

  ClassDef(VMCReplay, 1); // needed as long we inherit from TObject
};

#endif //O2_VMCREPLAY_H
