// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChipSegmentation.h
/// \brief Chip (sensor) segmentation description
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_CHIPSEGMENTATION_H_
#define ALICEO2_MFT_CHIPSEGMENTATION_H_

#include "MFTBase/VSegmentation.h"

namespace o2
{
namespace mft
{

class ChipSegmentation : public VSegmentation
{

 public:
  ChipSegmentation();
  ChipSegmentation(UInt_t uniqueID);

  ~ChipSegmentation() override = default;
  void Clear(const Option_t* /*opt*/) override { ; }
  void print(Option_t* /*option*/);

 private:
  ClassDefOverride(ChipSegmentation, 1);
};
} // namespace mft
} // namespace o2

#endif
