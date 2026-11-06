// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_VMC_SEEDERSERVICE_H
#define ALICEO2_VMC_SEEDERSERVICE_H

class TVirtualMC;
class TRandom;

#include <atomic>
#include <functional>

namespace o2
{
namespace base
{

// A simple (singleton) service to propagate random number seeding
// to VMC engines in a loosely-coupled, just-in-time way (without needing to compile against these libraries).
class VMCSeederService
{

 public:
  static VMCSeederService const& instance()
  {
    static VMCSeederService inst;
    return inst;
  }

  void setSeed() const; // will propagate the seed of o2::base::simRandom() to the VMC engine of this thread

  /// how often a seed was propagated; lets callers detect a silent no-op
  unsigned long long getSeedCount() const { return mSeedCount; }

  typedef std::function<void(TRandom*)> SeederFcn;

  /// installs the function that seeds Geant4 from a generator; called by the Geant4 configuration, which links Geant4
  static void setGeant4Seeder(SeederFcn fcn) { geant4Seeder() = std::move(fcn); }

 private:
  VMCSeederService();
  static SeederFcn& geant4Seeder()
  {
    static SeederFcn fcn;
    return fcn;
  }
  void initSeederFunction(TVirtualMC const*);

  SeederFcn mSeederFcn;                                  // the function talking to the VMC engine
  mutable std::atomic<unsigned long long> mSeedCount{0}; // number of setSeed() calls
};

} // namespace base
} // namespace o2

#endif
