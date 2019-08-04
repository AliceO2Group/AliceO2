// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ContainerFactory.h
/// \brief Definition of the ContainerFactory class

#ifndef ALICEO2_ITS_CONTAINERFACTORY_H_
#define ALICEO2_ITS_CONTAINERFACTORY_H_

#include "FairContFact.h" // for FairContFact, FairContainer (ptr only)
#include "Rtypes.h"       // for ContainerFactory::Class, ClassDef, etc

class FairParSet;

class FairContainer;

namespace o2
{
namespace its
{
class ContainerFactory : public FairContFact
{
 private:
  /// Creates the Container objects with all accepted
  /// contexts and adds them to
  /// the list of containers for the O2its library.
  void mSetAllContainers();

 public:
  /// Default constructor
  ContainerFactory();

  /// Default destructor
  ~ContainerFactory() override = default;
  /// Calls the constructor of the corresponding parameter container.
  /// For an actual context, which is not an empty string and not
  /// the default context
  /// of this container, the name is concatinated with the context.
  FairParSet* createContainer(FairContainer*) override;

  ClassDefOverride(ContainerFactory, 0); // Factory for all AliceO2 ITS parameter containers
};
} // namespace its
} // namespace o2

#endif
