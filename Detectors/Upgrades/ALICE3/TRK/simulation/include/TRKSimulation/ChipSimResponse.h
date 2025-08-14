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

#ifndef ALICEO2_TRKSIMULATION_CHIPSIMRESPONSE_H
#define ALICEO2_TRKSIMULATION_CHIPSIMRESPONSE_H

#include "ITSMFTSimulation/AlpideSimResponse.h"

namespace o2
{
namespace trk
{

class ChipSimResponse : public o2::itsmft::AlpideSimResponse
{
 public:
  ChipSimResponse() = default;
  ChipSimResponse(const ChipSimResponse& other) = default;
  ChipSimResponse(const o2::itsmft::AlpideSimResponse* base) : o2::itsmft::AlpideSimResponse(*base) {}

  void initData(int tableNumber, std::string dataPath, const bool quiet = true);

  ClassDef(ChipSimResponse, 1);
};

} // namespace trk
} // namespace o2

#endif // ALICEO2_TRKSIMULATION_CHIPSIMRESPONSE_H
