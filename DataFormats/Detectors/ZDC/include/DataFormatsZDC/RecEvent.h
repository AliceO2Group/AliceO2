// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_RECEVENT_H_
#define _ZDC_RECEVENT_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "MathUtils/Cartesian2D.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file RecEvent.h
/// \brief Class to describe reconstructed ZDC event (single BC with signal in one of detectors)
/// \author cortese@to.infn.it, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{

struct RecEvent {
  using TDCChannel = std::array<float, MaxTDCValues>;

  o2::InteractionRecord ir;
  uint32_t flags;                                   /// reconstruction flags
  std::array<float, NChannelsZEM> eneregyZEM;       /// signal in the electromagnetic ZDCs
  std::array<float, NChannelsZN> energyZNA;         /// reco E in 5 ZNA sectors + sum
  std::array<float, NChannelsZN> energyZNC;         /// reco E in 5 ZNC sectors + sum
  std::array<float, NChannelsZP> energyZPA;         /// reco E in 5 ZPA sectors + sum
  std::array<float, NChannelsZP> energyZPC;         /// reco E in 5 ZPC sectors + sum
  Point2D<float> centroidZNA;                       /// centroid coordinates for ZNA
  Point2D<float> centroidZNC;                       /// centroid coordinates for ZNC
  std::array<TDCChannel, NTDCChannels> tdcChannels; /// At most MaxTDCValues Values in ns per TDC channel

  void print() const;

  ClassDefNV(RecEvent, 1);
};

} // namespace zdc

/// Defining RecEvent explicitly as messageable
///
/// It does not fulfill is_messageable because the underlying ROOT
/// classes of Point2D are note trivially copyable.
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::zdc::RecEvent> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif
