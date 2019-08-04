// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CONTAINERFACTORY_H_
#define ALICEO2_TPC_CONTAINERFACTORY_H_

#include "FairContFact.h" // for FairContFact, FairContainer (ptr only)
#include "Rtypes.h"       // for ContainerFactory::Class, ClassDef, etc
class FairParSet;

class FairContainer;

namespace o2
{
namespace tpc
{

class ContainerFactory : public FairContFact
{
 private:
  void setAllContainers();

 public:
  ContainerFactory();
  ~ContainerFactory() override = default;
  FairParSet* createContainer(FairContainer*) override;
  ClassDefOverride(o2::tpc::ContainerFactory, 0); // Factory for all tpc parameter containers
};
} // namespace tpc
} // namespace o2
#endif
