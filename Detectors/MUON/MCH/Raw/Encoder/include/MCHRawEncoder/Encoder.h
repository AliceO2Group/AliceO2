// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_H
#define O2_MCH_RAW_ENCODER_H

#include <memory>
#include <set>
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawCommon/SampaCluster.h"
#include <functional>
#include <optional>

namespace o2::mch::raw
{

/// @brief An Encoder builds MCH raw data
///
/// Data is added using the addChannelData() method.
/// Then it can be exported using the moveToBuffer() method.
/// \nosubgrouping

class Encoder
{
 public:
  virtual ~Encoder() {}

  /** @name Main interface.
    */
  ///@{
  /// add data for one channel.
  ///
  /// \param dsId is the (electronic) identifier of a dual sampa
  /// \param chId dual sampa channel id 0..63
  /// \param data the actual data to be added
  virtual void addChannelData(DsElecId dsId, uint8_t chId, const std::vector<SampaCluster>& data) = 0;

  // startHeartbeatFrame sets the trigger (orbit,bunchCrossing) to be used
  // for all generated payload headers (until next call to this method).
  // Causes the alignment of the underlying gbts.
  virtual void startHeartbeatFrame(uint32_t orbit, uint16_t bunchCrossing) = 0;

  /// Export our encoded data.
  ///
  /// The internal words that have been accumulated so far are
  /// _moved_ (i.e. deleted from this object) to the external buffer of bytes
  /// Returns the number of bytes added to buffer.
  virtual size_t moveToBuffer(std::vector<uint8_t>& buffer) = 0;
  ///@}
};

/// createEncoder creates an encoder
///
/// template parameters :
///
/// \tparam FORMAT defines the data format (either BareFormat or UserLogic)
/// \tparam CHARGESUM defines the data format mode (either SampleMode or ChargeSumMode)
/// \tparam forceNoPhase to be deprecated ?
///
template <typename FORMAT, typename CHARGESUM, bool forceNoPhase = false>
std::unique_ptr<Encoder> createEncoder();
} // namespace o2::mch::raw
#endif
