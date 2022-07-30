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
#ifndef ALICEO2_EMCAL_CHANNEL_H
#define ALICEO2_EMCAL_CHANNEL_H

#include <cstdint>
#include <exception>
#include <vector>
#include "Rtypes.h"
#include "EMCALReconstruction/Bunch.h"

namespace o2
{

namespace emcal
{

/// \class Channel
/// \brief ALTRO channel representation
/// \ingroup EMCALreconstruction
///
/// The channel contains information about
/// a hardware channel in the raw stream. Those
/// information are:
/// - Hardware address
/// - Size of the payload of all bunches in the channel
///   as total number of 10-bit words
/// - Channel status (good or bad)
/// In addition it contains the data of all bunches in the
/// raw stream.
///
/// The hardware address itself encods
/// - Branch ID (bit 12)
/// - FEC ID (bits 7-10)
/// - ALTRO ID (bits 4-6)
/// - Channel ID (bits 0-3)
class Channel
{
 public:
  /// \class HardwareAddressError
  /// \brief Handling of uninitialized hardware addresses
  class HardwareAddressError : public std::exception
  {
   public:
    /// \brief Constructor
    HardwareAddressError() = default;

    /// \brief Destructor
    ~HardwareAddressError() noexcept override = default;

    /// \brief Access to error message
    /// \return error message
    const char* what() const noexcept override
    {
      return "Hardware address not initialized";
    }
  };

  /// \brief Dummy constructor
  Channel() = default;

  /// \brief Constructor initializing hardware address and payload size
  /// \param hardwareAddress Harware address
  /// \param payloadSize Size of the payload
  Channel(int32_t hardwareAddress, uint8_t payloadSize) : mHardwareAddress(hardwareAddress),
                                                          mPayloadSize(payloadSize),
                                                          mBunches()
  {
  }

  /// \brief Destructor
  ~Channel() = default;

  /// \brief Check whether the channel is bad
  /// \return true if the channel is bad, false otherwise
  bool isBadChannel() const { return mBadChannel; }

  /// \brief Get the full hardware address
  /// \return Hardware address
  ///
  /// The hardware address contains:
  /// - Branch ID (bit 12)
  /// - FEC ID (bits 7-10)
  /// - ALTRO ID (bits 4-6)
  /// - Channel ID (bits 0-3)
  uint16_t getHardwareAddress() const { return mHardwareAddress; }

  /// \brief Get the size of the payload
  /// \return Size of the payload as number of 10-bit samples (1/3rd words)
  uint16_t getPayloadSize() const { return mPayloadSize; }

  /// \brief Get list of bunches in the channel
  /// \return List of bunches
  const std::vector<Bunch>& getBunches() const { return mBunches; }

  /// \brief Provide the branch index for the current hardware address
  /// \return RCU branch index (0 or 1)
  /// \throw HadrwareAddressError in case the hardware address is not initialized
  int getBranchIndex() const;

  /// \brief Provide the front-end card index (0-9) in branch for the current hardware address
  /// \return Front-end card index for the current hardware address
  /// \throw HadrwareAddressError in case the hardware address is not initialized
  int getFECIndex() const;

  /// \brief Provide the altro chip index for the current hardware address
  /// \return Altro chip index for the current hardware address
  /// \throw HadrwareAddressError in case the hardware address is not initialized
  int getAltroIndex() const;

  /// \brief Provide the channel index for the current hardware address
  /// \return Channel index for the current hardware address
  /// \throw HadrwareAddressError in case the hardware address is not initialized
  int getChannelIndex() const;

  /// \brief Add bunch to the channel
  /// \param bunch Bunch to be added
  ///
  /// This function will copy the bunch information to the
  /// object, which might be expensive. Better use the
  /// function createBunch.
  void addBunch(const Bunch& bunch) { mBunches.emplace_back(bunch); }

  /// \brief Set the hardware address
  /// \param hardwareAddress Hardware address
  void setHardwareAddress(uint16_t hardwareAddress) { mHardwareAddress = hardwareAddress; }

  /// \brief Set the size of the payload in number of 10-bit words
  /// \param payloadSize Size of the payload
  void setPayloadSize(uint8_t payloadSize) { mPayloadSize = payloadSize; }

  /// \brief Mark the channel status
  /// \param badchannel Bad channel status (true if bad)
  void setBadChannel(bool badchannel) { mBadChannel = badchannel; }

  /// \brief Create and initialize a new bunch and return reference to it
  /// \param bunchlength Length of the bunch
  /// \param starttime Start time of the bunch
  Bunch& createBunch(uint8_t bunchlength, uint8_t starttime);

  /// \brief Extrcting hardware address from the channel header word
  /// \param channelheader Channel header word
  static int getHardwareAddressFromChannelHeader(int channelheader) { return channelheader & 0xFFF; };

  /// \brief Extrcting payload size from the channel header word
  /// \param channelheader Channel header word
  static int getPayloadSizeFromChannelHeader(int channelheader) { return (channelheader >> 16) & 0x3FF; }

  /// \brief Extracting branch index from the hardware address
  /// \param hwaddress Hardware address of the channel
  static int getBranchIndexFromHwAddress(int hwaddress) { return ((hwaddress >> 11) & 0x1); }

  /// \brief Extracting FEC index in branch from the hardware address
  /// \param hwaddress Hardware address of the channel
  static int getFecIndexFromHwAddress(int hwaddress) { return ((hwaddress >> 7) & 0xF); }

  /// \brief Extracting ALTRO index from the hardware address
  /// \param hwaddress Hardware address of the channel
  static int getAltroIndexFromHwAddress(int hwaddress) { return ((hwaddress >> 4) & 0x7); }

  /// \brief Extracting Channel index in FEC from the hardware address
  /// \param hwaddress Hardware address of the channel
  static int getChannelIndexFromHwAddress(int hwaddress) { return (hwaddress & 0xF); }

 private:
  int32_t mHardwareAddress = -1; ///< Hardware address
  uint16_t mPayloadSize = 0;     ///< Payload size
  bool mBadChannel;              ///< Bad channel status
  std::vector<Bunch> mBunches;   ///< Bunches in channel;

  ClassDefNV(Channel, 1);
};

} // namespace emcal
} // namespace o2

#endif