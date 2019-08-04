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
// -----                    Magnet  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------

#ifndef ALICEO2_PASSIVE_MAGNET_H
#define ALICEO2_PASSIVE_MAGNET_H

#include "FairModule.h" // for FairModule
#include "Rtypes.h"     // for Magnet::Class, Bool_t, etc

namespace o2
{
namespace passive
{
class Magnet : public FairModule
{
 public:
  Magnet(const char* name, const char* Title = "ALICE Magnet");
  Magnet();
  ~Magnet() override;
  void ConstructGeometry() override;
  void createMaterials();

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

 private:
  Magnet(const Magnet& orig);
  Magnet& operator=(const Magnet&);

  ClassDefOverride(o2::passive::Magnet, 1);
};
} // namespace passive
} // namespace o2

#endif // MAGNET_H
