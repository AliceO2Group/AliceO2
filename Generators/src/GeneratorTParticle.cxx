// Copyright 2023 CERN and copyright holders of ALICE O2.
//
// See https://alice-o2.web.cern.ch/copyright for details of the
// copyright holders.  All rights not expressly granted are reserved.
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
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TChain.h>
#include <TClonesArray.h>
#include <TParticle.h>
#include <filesystem>
#include <thread>
#include <sstream>
#include <cstdlib>
#include <sys/types.h> // POSIX only
#include <sys/stat.h>  // POISX only
#include <cstdio>

namespace o2
{
namespace eventgen
{
GeneratorTParticle::~GeneratorTParticle()
{
  if (mChain) {
    TFile* file = mChain->GetCurrentFile();
    if (file)
      mChain->RecursiveRemove(file);
    delete mChain;
  }
  if (mProgCmd.empty())
    return;

  // Get the file we're reading from
  std::filesystem::path p(mFileNames.front());

  // Wait until child process creates the file
  if (not std::filesystem::exists(p))
    return;

  // Remove temporary file
  std::error_code ec;
  std::filesystem::remove(p, ec);
}

void GeneratorTParticle::setFileNames(const std::string& val)
{
  std::stringstream s;
  std::string f;
  while (std::getline(s, f, ','))
    mFileNames.push_back(f);
}

Bool_t GeneratorTParticle::Init()
{
  mChain = new TChain(mTreeName.c_str());
  mTParticles = new TClonesArray("TParticle");
  mChain->SetBranchAddress(mBranchName.c_str(), &mTParticles);

  if (not mProgCmd.empty()) {
    // Set filename to be a temporary name
    // Should perhaps use
    //
    //   TString base("xxxxxx");
    //   auto fp = gSystem->TempFileName(base);
    //   fclose(fp);
    //
    std::string filename = std::tmpnam(nullptr);

    // Build command line, Assumes command line parameter
    //
    //  -n NUMBER   of events to produce
    //  -o FILENAME of output file
    //
    // A script can be wrapped around existing EGs to ensure these
    // options are observed.
    std::string cmd =
      mProgCmd +
      " -n " + std::to_string(mNEvents) +
      " -o " + filename + " &";
    LOG(info) << "EG command line is \"" << cmd << "\"";

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(fatal) << "Failed to spawn \"" << cmd << "\"";
      return false;
    }

    mFileNames.clear();
    mFileNames.push_back(filename);
  }
  for (auto filename : mFileNames)
    mChain->AddFile(filename.c_str());

  return true;
}

void GeneratorTParticle::waitForData()
{
  if (mProgCmd.empty())
    return; // Not from child process

  using namespace std::chrono_literals;

  // Get the file we're reading from
  std::filesystem::path p(mFileNames.front());

  LOG(info) << "Waiting for data on " << p;

  // Wait until child process creates the file
  while (not std::filesystem::exists(p))
    std::this_thread::sleep_for(500ms);

  // Wait until we have more data in the file than just the file
  // header
  while (std::filesystem::file_size(p) <= 256)
    std::this_thread::sleep_for(500ms);

  // Give the child process 1 second to post the data to the file
  LOG(info) << "Got data in " << p << ", sleeping for a while";
  std::this_thread::sleep_for(1s);
}

Bool_t GeneratorTParticle::generateEvent()
{
  if (mEntry == 0)
    waitForData();

  int read = mChain->GetEntry(mEntry);
  mEntry++;

  if (read < 0)
    LOG(error) << "Failed to read entry " << mEntry << " of chain";

  if (read <= 0)
    return false;

  return true;
}

Bool_t GeneratorTParticle::importParticles()
{
  for (auto* object : *mTParticles) {
    TParticle* particle = static_cast<TParticle*>(object);
    auto statusCode = particle->GetStatusCode();
    if (!mcgenstatus::isEncoded(statusCode))
      statusCode = mcgenstatus::MCGenStatusEncoding(statusCode, 0)
                     .fullEncoding;

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
