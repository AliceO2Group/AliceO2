// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliPassiveContFact  file                    -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------

#ifndef ALICEO2_PASSIVE_CONTFACT_H
#define ALICEO2_PASSIVE_CONTFACT_H

#include "FairContFact.h" // for FairContFact, etc
#include "Rtypes.h"       // for AliPassiveContFact::Class, etc

class FairParSet;

namespace o2
{
namespace passive
{

class PassiveContFact : public FairContFact
{
 private:
  void setAllContainers();

 public:
  PassiveContFact();
  ~PassiveContFact() override { ; }
  FairParSet* createContainer(FairContainer*) override;
  ClassDefOverride(o2::passive::PassiveContFact, 0); // Factory for all Passive parameter containers
};
} // namespace passive
} // namespace o2
#endif /* !PNDPASSIVECONTFACT_H */
