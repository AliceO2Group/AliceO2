// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMSETUP_G4LOCALFIELDCONSTRUCTION_H_
#define O2_SIMSETUP_G4LOCALFIELDCONSTRUCTION_H_

#include "TG4VUserPostDetConstruction.h"

#include <memory>

namespace o2::g4config
{

/// Gives every volume with its own field parameters (/mcDet/createMagFieldParameters <vol>)
/// a local copy of the global field, so that the /mcMagField/<vol>/ settings take effect.
class G4LocalFieldConstruction : public TG4VUserPostDetConstruction
{
 public:
  /// next is another construction step run first (may be nullptr); it is owned
  explicit G4LocalFieldConstruction(TG4VUserPostDetConstruction* next) : mNext(next) {}
  void Construct() override;

 private:
  std::unique_ptr<TG4VUserPostDetConstruction> mNext;
};

} // namespace o2::g4config

#endif // O2_SIMSETUP_G4LOCALFIELDCONSTRUCTION_H_
