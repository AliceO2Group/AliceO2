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
#include "DataFormatsZDC/BCRecData.h"
#include "DataFormatsZDC/ZDCEnergy.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "MathUtils/Cartesian.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>
#include <vector>
#include <map>

/// \file RecEvent.h
/// \brief Class to describe reconstructed ZDC event (single BC with signal in one of detectors)
/// \author cortese@to.infn.it, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{
struct RecEvent {
  std::vector<o2::zdc::BCRecData> mRecBC;
  std::vector<o2::zdc::ZDCEnergy> mEnergy;
  std::vector<o2::zdc::ZDCTDCData> mTDCData;
  std::vector<uint16_t> mInfo;
  // Add new bunch crossing without data
  void addBC(o2::InteractionRecord ir)
  {
    //mRecBC.emplace_back(mEnergy.size(), mTDCData.size(), mInfo.size(), ir);
    mRecBC.emplace_back(mEnergy.size(), 0, mInfo.size(), ir);
  }
  void addEnergy(uint8_t ch, float energy)
  {
    mEnergy.emplace_back(ch, energy);
    mRecBC.back().addEnergy();
  }
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
