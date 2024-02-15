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

#ifndef MI3LAYER_H_
#define MI3LAYER_H_

#include <string>
#include <vector>

namespace o2
{
namespace mi3
{
class MIDLayer
{
  class Module
  {
   public:
    Module(std::string moduleName);

   private:
    std::string mName;
  };

  class Stave
  {
   public:
    Stave(std::string staveName);

   private:
    std::string mName;
    std::vector<Module> mModules;
  };

 public:
  MIDLayer();

 private:
  std::string mName;
  std::vector<Stave> mStaves;
};
} // namespace mi3
} // namespace o2

#endif // MI3LAYER_H