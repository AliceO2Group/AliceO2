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

  void setSeed() const; // will propagate seed to the VMC engines

  typedef std::function<void()> SeederFcn;

 private:
  VMCSeederService();
  void initSeederFunction(TVirtualMC const*);

  SeederFcn mSeederFcn; // the just-in-time compiled function talking to the VMC engines
};

} // namespace base
} // namespace o2

#endif
