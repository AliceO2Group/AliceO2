// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "BareElinkEncoder.h"

namespace o2::mch::raw
{

template <>
void ElinkEncoder<BareFormat, ChargeSumMode>::appendCharges(const SampaCluster& sc)
{
  append20(sc.chargeSum);
}

template <>
void ElinkEncoder<BareFormat, SampleMode>::appendCharges(const SampaCluster& sc)
{
  for (auto& s : sc.samples) {
    append10(s);
  }
}

} // namespace o2::mch::raw
