// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimulationDataFormat/DigitizationContext.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include <TChain.h>
#include <iostream>

using namespace o2::steer;

void DigitizationContext::printCollisionSummary() const
{
  std::cout << "Summary of DigitizationContext --\n";
  std::cout << "Parts per collision " << mMaxPartNumber << "\n";
  std::cout << "Collision parts taken from simulations specified by prefix:\n";
  for (int p = 0; p < mSimPrefixes.size(); ++p) {
    std::cout << "Part " << p << " : " << mSimPrefixes[p] << "\n";
  }
  std::cout << "Number of Collisions " << mEventRecords.size() << "\n";
  for (int i = 0; i < mEventRecords.size(); ++i) {
    std::cout << "Collision " << i << " TIME " << mEventRecords[i].timeNS;
    for (auto& e : mEventParts[i]) {
      std::cout << " (" << e.sourceID << " , " << e.entryID << ")";
    }
    std::cout << "\n";
  }
}

void DigitizationContext::setSimPrefixes(std::vector<std::string> const& prefixes)
{
  mSimPrefixes = prefixes;
  // the number should correspond to the number of parts
  if (mSimPrefixes.size() != mMaxPartNumber) {
    std::cerr << "Inconsistent number of simulation prefixes and part numbers";
  }
}

bool DigitizationContext::initSimChains(o2::detectors::DetID detid, std::vector<TChain*>& simchains) const
{
  if (!(simchains.size() == 0)) {
    // nothing to do ... already setup
    return false;
  }

  simchains.emplace_back(new TChain("o2sim"));
  // add the main (background) file
  simchains.back()->AddFile(o2::base::NameConf::getHitsFileName(detid, mSimPrefixes[0].data()).c_str());

  for (int source = 1; source < mSimPrefixes.size(); ++source) {
    simchains.emplace_back(new TChain("o2sim"));
    // add signal files
    simchains.back()->AddFile(o2::base::NameConf::getHitsFileName(detid, mSimPrefixes[source].data()).c_str());
  }
  return true;
}

o2::parameters::GRPObject const& DigitizationContext::getGRP() const
{
  if (!mGRP) {
    // we take the GRP from the background file
    // maybe we should add a check that all GRPs are consistent ..
    mGRP = o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName(mSimPrefixes[0].data()).c_str());
  }
  return *mGRP;
}
