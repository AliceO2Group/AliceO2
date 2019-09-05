// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CustomStreamers.cxx
/// \brief Custom streamer implementations for module SimulationDataFormat
/// \author Matthias Richter (Matthias.Richter@scieq.net)
/// \since Sep 5 2019

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TBuffer.h"

namespace o2
{
namespace dataformats
{

template <>
void MCTruthContainer<MCCompLabel>::Streamer(TBuffer& R__b)
{
  // the custom streamer for MCTruthContainer<MCCompLabel>

  if (R__b.IsReading()) {
    R__b.ReadClassBuffer(MCTruthContainer<MCCompLabel>::Class(), this);
    inflate();
  } else {
    deflate();
    R__b.WriteClassBuffer(MCTruthContainer<MCCompLabel>::Class(), this);
    inflate();
  }
}

} // namespace dataformats
} // namespace o2
