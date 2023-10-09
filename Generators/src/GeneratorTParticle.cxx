// Copyright 2023-2099 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Christian Holm Christensen <cholm@nbi.dk>
#include <Generators/GeneratorTParticle.h>
#include <Generators/GeneratorTParticleParam.h>
#include <SimulationDataFormat/MCGenProperties.h>
#include <SimConfig/SimConfig.h>
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TChain.h>
#include <TClonesArray.h>
#include <TParticle.h>

namespace o2
{
namespace eventgen
{
/*****************************************************************/
GeneratorTParticle::GeneratorTParticle()
{
  setOutputSwitch("-o");
}

/*****************************************************************/
GeneratorTParticle::~GeneratorTParticle()
{
  if (mChain) {
    TFile* file = mChain->GetCurrentFile();
    if (file) {
      mChain->RecursiveRemove(file);
    }
    delete mChain;
  }
  if (mCmd.empty()) {
    return;
  }

  removeTemp();
}
/*****************************************************************/
Bool_t GeneratorTParticle::Init()
{
  mChain = new TChain(mTreeName.c_str());
  mTParticles = new TClonesArray("TParticle");
  mChain->SetBranchAddress(mBranchName.c_str(), &mTParticles);

  if (not mCmd.empty()) {
    // Set filename to be a temporary name
    if (not makeTemp()) {
      return false;
    }

    // Build command line, Assumes command line parameter
    std::string cmd = makeCmdLine();
    LOG(info) << "EG command line is \"" << cmd << "\"";

    // Execute the background command
    if (not executeCmdLine(cmd)) {
      LOG(fatal) << "Failed to spawn \"" << cmd << "\"";
      return false;
    }
  }
  for (auto filename : mFileNames) {
    mChain->AddFile(filename.c_str());
  }

  // Clear the array of file names
  mFileNames.clear();

  return true;
}

/*****************************************************************/
void GeneratorTParticle::setup(const GeneratorFileOrCmdParam& param0,
                               const GeneratorTParticleParam& param,
                               const conf::SimConfig& config)
{
  GeneratorFileOrCmd::setup(param0, config);
  setTreeName(param.treeName);
  setBranchName(param.branchName);
}

/*****************************************************************/
Bool_t GeneratorTParticle::generateEvent()
{
  // If this is the first entry, and we're executing a command, then
  // wait until the input file exists and actually contain some data.
  if (mEntry == 0 and not mCmd.empty()) {
    waitForData(mTemporary);
  }

  // Read in the next entry in the chain
  int read = mChain->GetEntry(mEntry);
  mEntry++;

  // If we got an error while reading, then give error message
  if (read < 0) {
    LOG(error) << "Failed to read entry " << mEntry << " of chain";
  }

  // If we had an error or nothing was read back, then return false
  if (read <= 0) {
    return false;
  }

  return true;
}

Bool_t GeneratorTParticle::importParticles()
{
  for (auto* object : *mTParticles) {
    TParticle* particle = static_cast<TParticle*>(object);
    auto statusCode = particle->GetStatusCode();
    if (!mcgenstatus::isEncoded(statusCode)) {
      statusCode = mcgenstatus::MCGenStatusEncoding(statusCode, 0)
                     .fullEncoding;
    }

    mParticles.emplace_back(particle->GetPdgCode(),
                            statusCode,
                            particle->GetFirstMother(),
                            particle->GetSecondMother(),
                            particle->GetFirstDaughter(),
                            particle->GetLastDaughter(),
                            particle->Px(),
                            particle->Py(),
                            particle->Pz(),
                            particle->Energy(),
                            particle->Vx(),
                            particle->Vy(),
                            particle->Vz(),
                            particle->T());
    auto& tgt = mParticles[mParticles.size() - 1];
    tgt.SetPolarTheta(particle->GetPolarTheta());
    tgt.SetPolarPhi(particle->GetPolarPhi());
    tgt.SetCalcMass(particle->GetCalcMass());
    tgt.SetWeight(particle->GetWeight());
  }
  return true;
}
} // namespace eventgen
} // namespace o2
//
// EOF
//
