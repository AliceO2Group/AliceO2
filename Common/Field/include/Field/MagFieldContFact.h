// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagFieldContFact.h
/// \brief Definition of the MagFieldContFact: factory for ALICE mag. field
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGFIELDCONTFACT_
#define ALICEO2_FIELD_MAGFIELDCONTFACT_

#include "FairContFact.h"
#include "Rtypes.h" // for ClassDef

class FairParSet;

namespace o2
{
namespace field
{

class MagFieldParam;

class MagFieldContFact : public FairContFact
{
 private:
  void setAllContainers();

 public:
  MagFieldContFact();
  ~MagFieldContFact() override = default;
  FairParSet* createContainer(FairContainer*) override;

  ClassDefOverride(MagFieldContFact, 0); // Factory for Magnetic field parameters containers
};

} // namespace field
} // namespace o2
#endif
