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
#include "SimulationDataFormat/MCEventHeader.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include <TChain.h>
#include <TFile.h>
#include <iostream>
#include <MathUtils/Cartesian3D.h>

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
    std::cout << "Collision " << i << " TIME " << mEventRecords[i];
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

/// Common functions the setup input TChains for reading kinematics information, given the state (prefixes) encapsulated
/// by this context. The input vector needs to be empty otherwise nothing will be done.
/// return boolean saying if input simchains was modified or not
bool DigitizationContext::initSimKinematicsChains(std::vector<TChain*>& simkinematicschains) const
{
  if (!(simkinematicschains.size() == 0)) {
    // nothing to do ... already setup
    return false;
  }

  simkinematicschains.emplace_back(new TChain("o2sim"));
  // add the main (background) file
  simkinematicschains.back()->AddFile(o2::base::NameConf::getMCKinematicsFileName(mSimPrefixes[0].data()).c_str());

  for (int source = 1; source < mSimPrefixes.size(); ++source) {
    simkinematicschains.emplace_back(new TChain("o2sim"));
    // add signal files
    simkinematicschains.back()->AddFile(o2::base::NameConf::getMCKinematicsFileName(mSimPrefixes[source].data()).c_str());
  }
  return true;
}

bool DigitizationContext::checkVertexCompatibility(bool verbose) const
{
  if (mMaxPartNumber == 1) {
    return true;
  }

  auto checkVertexPair = [](Point3D<double> const& p1, Point3D<double> const& p2) -> bool {
    return (p2 - p1).Mag2() < 1E-6;
  };

  std::vector<TChain*> kinematicschain;
  std::vector<TBranch*> headerbranches;
  std::vector<o2::dataformats::MCEventHeader*> headers;
  std::vector<Point3D<double>> vertices;
  initSimKinematicsChains(kinematicschain);
  bool consistent = true;
  if (kinematicschain.size() > 0) {
    headerbranches.resize(kinematicschain.size(), nullptr);
    headers.resize(kinematicschain.size(), nullptr);
    // loop over all collisions in this context
    int collisionID = 0;
    for (auto& collision : getEventParts()) {
      collisionID++;
      vertices.clear();
      for (auto& part : collision) {
        const auto source = part.sourceID;
        const auto entry = part.entryID;
        auto chain = kinematicschain[source];
        if (!headerbranches[source]) {
          headerbranches[source] = chain->GetBranch("MCEventHeader.");
          headerbranches[source]->SetAddress(&headers[source]);
        }
        // get the MCEventHeader to read out the vertex
        headerbranches[source]->GetEntry(entry);
        auto header = headers[source];
        vertices.emplace_back(header->GetX(), header->GetY(), header->GetZ());
      }
      // analyse vertex matching
      bool thiscollision = true;
      const auto& p1 = vertices[0];
      for (int j = 1; j < vertices.size(); ++j) {
        const auto& p2 = vertices[j];
        bool thischeck = checkVertexPair(p1, p2);
        thiscollision &= thischeck;
      }
      if (verbose && !thiscollision) {
        std::stringstream text;
        text << "Found inconsistent vertices for digit collision ";
        text << collisionID << " : ";
        for (auto& p : vertices) {
          text << p << " ";
        }
        LOG(ERROR) << text.str();
      }
      consistent &= thiscollision;
    }
  }
  return consistent;
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

void DigitizationContext::saveToFile(std::string_view filename) const
{
  TFile file(filename.data(), "RECREATE");
  auto cl = TClass::GetClass(typeid(*this));
  file.WriteObjectAny(this, cl, "DigitizationContext");
  file.Close();
}

DigitizationContext const* DigitizationContext::loadFromFile(std::string_view filename)
{
  DigitizationContext* incontext = nullptr;
  TFile file(filename.data(), "OPEN");
  file.GetObject("DigitizationContext", incontext);
  return incontext;
}
