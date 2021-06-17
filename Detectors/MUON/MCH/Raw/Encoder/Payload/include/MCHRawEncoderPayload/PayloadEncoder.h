// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_PAYLOAD_ENCODER_H
#define O2_MCH_RAW_PAYLOAD_ENCODER_H

#include <memory>
#include <set>
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawElecMap/Mapper.h"
#include <functional>
#include <optional>
#include <set>

namespace o2::mch::raw
{

/// @brief A PayloadEncoder builds MCH raw data (payload part only)
///
/// Data is added using the addChannelData() method.
/// Then it can be exported using the moveToBuffer() method.
/// The output buffer contains pairs of (DataBlockHeader,payload)
///
/// \nosubgrouping

class PayloadEncoder
{
 public:
  virtual ~PayloadEncoder() = default;

  /** @name Main interface.
    */
  ///@{
  /// add data for one channel.
  ///
  /// \param dsId is the (electronic) identifier of a dual sampa
  /// \param chId dual sampa channel id 0..63
  /// \param data the actual data to be added
  virtual void addChannelData(DsElecId dsId, DualSampaChannelId chId,
                              const std::vector<SampaCluster>& data) = 0;

  // startHeartbeatFrame sets the trigger (orbit,bunchCrossing) to be used
  // for all generated payload headers (until next call to this method).
  // Causes the alignment of the underlying gbts.
  virtual void startHeartbeatFrame(uint32_t orbit, uint16_t bunchCrossing) = 0;

  // addHeartbeatHeaders adds Heartbeat Sampa headers for each dual sampa in the
  // given set.
  virtual void addHeartbeatHeaders(const std::set<DsElecId>& dsids) = 0;

  /// Export our encoded data.
  ///
  /// The internal words that have been accumulated so far are
  /// _moved_ (i.e. deleted from this object) to the external buffer of bytes
  /// Returns the number of bytes added to buffer.
  virtual size_t moveToBuffer(std::vector<std::byte>& buffer) = 0;
  ///@}
};

/// createPayloadEncoder creates a payload encoder
///
/// \param userLogic whether to encode in UserLogic (true) or BareFormat (false)
/// \param version defines the version of the encoding format
///         (currently 0 or 1 are possible, just for UserLogic)
/// \param chargeSumMode whether to encode in charge sum mode (true) or sample mode (false)
/// \param solar2feelink a mapper to convert solarId values into FeeLinkId objects
std::unique_ptr<PayloadEncoder> createPayloadEncoder(Solar2FeeLinkMapper solar2feelink,
                                                     bool userLogic,
                                                     int version,
                                                     bool chargeSumMode);

} // namespace o2::mch::raw
#endif
