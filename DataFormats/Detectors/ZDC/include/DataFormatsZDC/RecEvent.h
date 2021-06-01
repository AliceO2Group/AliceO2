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
#include "ZDCBase/Constants.h"
#include "MathUtils/Cartesian.h"
#include <Rtypes.h>
#include <array>
#include <vector>
#include <map>

/// \file RecEvent.h
/// \brief Class to describe reconstructed ZDC event (single BC with signal in one of detectors)
/// \author pietro.cortese@cern.ch, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{
struct RecEvent {
  std::vector<o2::zdc::BCRecData> mRecBC;    /// Interaction record and references to data
  std::vector<o2::zdc::ZDCEnergy> mEnergy;   /// ZDC energy
  std::vector<o2::zdc::ZDCTDCData> mTDCData; /// ZDC TDC
  std::vector<uint16_t> mInfo;               /// Event quality information
  // Add new bunch crossing without data
  inline void addBC(o2::InteractionRecord ir)
  {
    mRecBC.emplace_back(mEnergy.size(), mTDCData.size(), mInfo.size(), ir);
  }
  inline void addBC(o2::InteractionRecord ir, uint32_t channels, uint32_t triggers)
  {
    mRecBC.emplace_back(mEnergy.size(), mTDCData.size(), mInfo.size(), ir);
    mRecBC.back().channels = channels;
    mRecBC.back().triggers = triggers;
  }
  // Add energy
  inline void addEnergy(uint8_t ch, float energy)
  {
    mEnergy.emplace_back(ch, energy);
    mRecBC.back().addEnergy();
  }
  // Add TDC
  inline void addTDC(uint8_t ch, int16_t val, int16_t amp)
  {
    mTDCData.emplace_back(ch, val, amp);
    mRecBC.back().addTDC();
  }
  // Add event information
  inline void addInfo(uint8_t ch, uint16_t info)
  {
    if (ch >= NChannels) {
      ch = 0x1f;
    }
    info = (info & 0x07ff) || ch << 11;
    mInfo.emplace_back(info);
    mRecBC.back().addInfo();
  }
  void print() const;
  // TODO: remove persitency of this object (here for debugging)
  ClassDefNV(RecEvent, 1);
};

} // namespace zdc
} // namespace o2

#endif
