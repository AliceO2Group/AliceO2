// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FairLogger.h"
#include "CommonDataFormat/BunchFilling.h"
#include <TClass.h>
#include <TFile.h>

using namespace o2;

//_________________________________________________
void BunchFilling::setBC(int bcID, bool active)
{
  // add interacting BC slot
  if (bcID >= o2::constants::lhc::LHCMaxBunches) {
    LOG(FATAL) << "BCid is limited to " << 0 << '-' << o2::constants::lhc::LHCMaxBunches - 1;
  }
  mPattern.set(bcID, active);
}

//_________________________________________________
void BunchFilling::setBCTrain(int nBC, int bcSpacing, int firstBC)
{
  // add interacting BC train with given spacing starting at given place, i.e.
  // train with 25ns spacing should have bcSpacing = 1
  for (int i = 0; i < nBC; i++) {
    setBC(firstBC);
    firstBC += bcSpacing;
  }
}

//_________________________________________________
void BunchFilling::setBCTrains(int nTrains, int trainSpacingInBC, int nBC, int bcSpacing, int firstBC)
{
  // add nTrains trains of interacting BCs with bcSpacing within the train and trainSpacingInBC empty slots
  // between the trains
  for (int it = 0; it < nTrains; it++) {
    setBCTrain(nBC, bcSpacing, firstBC);
    firstBC += nBC * bcSpacing + trainSpacingInBC;
  }
}

//_________________________________________________
void BunchFilling::print(int bcPerLine) const
{
  bool endlOK = false;
  for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
    printf("%c", mPattern[i] ? '+' : '-');
    if (((i + 1) % bcPerLine) == 0) {
      printf("\n");
      endlOK = true;
    } else {
      endlOK = false;
    }
  }
  if (!endlOK) {
    printf("\n");
  }
}

//_______________________________________________
BunchFilling* BunchFilling::loadFrom(const std::string& fileName, const std::string& objName)
{
  // load object from file
  TFile fl(fileName.data());
  if (fl.IsZombie()) {
    LOG(ERROR) << "Failed to open " << fileName;
    return nullptr;
  }
  std::string nm = objName.empty() ? o2::BunchFilling::Class()->GetName() : objName;
  auto bf = reinterpret_cast<o2::BunchFilling*>(fl.GetObjectChecked(nm.c_str(), o2::BunchFilling::Class()));
  if (!bf) {
    LOG(ERROR) << "Did not find object named " << nm;
    return nullptr;
  }
  return bf;
}
