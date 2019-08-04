// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagFieldFact.h
/// \brief Definition of the MagFieldFact: factory for ALIDE mag. field from MagFieldParam
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGFIELDFACT_H
#define ALICEO2_FIELD_MAGFIELDFACT_H

#include "FairFieldFactory.h"
#include "Field/MagneticField.h"

class FairField;

namespace o2
{
namespace field
{
class MagFieldParam;

class MagFieldFact : public FairFieldFactory
{
 public:
  MagFieldFact();
  ~MagFieldFact() override;
  FairField* createFairField() override;
  void SetParm() override;

 private:
  MagFieldFact(const MagFieldFact&);
  MagFieldFact& operator=(const MagFieldFact&);

  MagFieldParam* mFieldPar;

  ClassDefOverride(MagFieldFact, 2);
};
} // namespace field
} // namespace o2

#endif
