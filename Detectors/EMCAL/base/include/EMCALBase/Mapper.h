// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef __O2_EMCAL_MAPPER_H__
#define __O2_EMCAL_MAPPER_H__

#include <cstdint>
#include <exception>
#include <functional>
#include <iosfwd>
#include <unordered_map>
#include <sstream>
#include <string>

#include "RStringView.h"
#include "Rtypes.h"
#include "DataFormatsEMCAL/Constants.h"

namespace o2
{

namespace emcal
{

/// \class Mapper
/// \brief ALTRO mapping for calorimeters
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Aug 19, 2019
///
/// Based on AliAltroMapping by C. Cheshkov and AliCaloAltroMapping by G. Balbastre
class Mapper
{
 public:
  /// \struct ChannelID
  /// \brief Mapped information of a channel
  struct ChannelID {
    uint8_t mRow;               ///< Row of the channel in module
    uint8_t mColumn;            ///< Column of the channel in module
    ChannelType_t mChannelType; ///< Type of the channel (see \ref o2::emcal::ChannelType for channel type definitions)

    bool operator==(const ChannelID& other) const
    {
      return mRow == other.mRow && mColumn == other.mColumn && mChannelType == other.mChannelType;
    }
    friend std::ostream& operator<<(std::ostream& stream, const Mapper::ChannelID& channel);
  };

  /// \struct ChannelIDHasher
  /// \brief Hash functor for channel ID
  struct ChannelIDHasher {

    /// \brief Functor implementation
    /// \param s Channel for which to determine a hash value
    /// \return hash value for channel ID
    size_t operator()(const ChannelID& s) const
    {
      size_t h1 = std::hash<int>()(s.mRow);
      size_t h2 = std::hash<int>()(s.mColumn);
      size_t h3 = std::hash<int>()(o2::emcal::channelTypeToInt(s.mChannelType));
      return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
      return h1 ^ (h2 << 1);
    }
  };

  /// \class AddressNotFoundException
  /// \brief Error handling requests for unknown hardware addresses
  class AddressNotFoundException : public std::exception
  {
   public:
    /// \brief Constructor initializing the exception
    /// \param address Hardware address raising the exception
    AddressNotFoundException(int address) : exception(),
                                            mAddress(address),
                                            mMessage("Hardware address " + std::to_string(address) + " not found")
    {
    }

    /// \brief Destructor
    ~AddressNotFoundException() noexcept override = default;

    /// \brief Access to error message
    /// \return Error message of the exception
    const char* what() const noexcept override { return mMessage.data(); }

    /// \brief Access to hardware address raising the exception
    /// \return Hardware address
    int getAddress() const noexcept { return mAddress; }

   private:
    int mAddress;         ///< Hardware address raising the exception
    std::string mMessage; ///< Message of the exception
  };

  /// \class Address range exception
  /// \brief Error handling hardware addresses outside range
  class AddressRangeException : public std::exception
  {
   public:
    /// \brief Constructor initializing the exception
    /// \param address Hardware address raising the exception
    /// \param maxaddress Maximum address in range
    AddressRangeException(int address, int maxaddress) : exception(),
                                                         mAddress(address),
                                                         mMaxAddress(maxaddress),
                                                         mMessage("Hardware (ALTRO) address (" + std::to_string(mAddress) + " outside the range (0 -> " + std::to_string(mMaxAddress) + ") !")
    {
    }

    /// \brief Destructor
    ~AddressRangeException() noexcept override = default;

    /// \brief Access to error message of the exception
    /// \return Error messages
    const char* what() const noexcept override { return mMessage.data(); }

    /// \brief Access to hardware address raising the exception
    /// \return Hardware address
    int getAddress() const noexcept { return mAddress; }

    /// \brief Access to max hardware address in mapping
    /// \return Hardware address
    int getMaxAddress() const noexcept { return mMaxAddress; }

   private:
    int mAddress;         ///< Address raising the exception
    int mMaxAddress;      ///< Max. hardware address in mapping
    std::string mMessage; ///< Message connected to the exception
  };

  /// \class ChannelNotFoundException
  /// \brief Exception handling invalid channel ID
  class ChannelNotFoundException : public std::exception
  {
   public:
    /// \brief Constructor initializing the exception
    /// \param id Channel ID rausing the exception
    ChannelNotFoundException(ChannelID id) : std::exception(),
                                             mChannelID(id),
                                             mMessage()
    {
      std::stringstream msgbuilder;
      msgbuilder << "Channel with " << mChannelID << " not found.";
      mMessage = msgbuilder.str();
    }

    /// \brief Destructor
    ~ChannelNotFoundException() noexcept override = default;

    /// \brief Access to error message of the exception
    /// \return Error message
    const char* what() const noexcept override
    {
      return mMessage.data();
    }

    /// \brief Access to channel ID raising the exception
    /// \return Channel ID raising the exception
    const ChannelID& getChannel() const { return mChannelID; }

   private:
    ChannelID mChannelID; ///< ChannelID raising the exception
    std::string mMessage; ///< Error message related
  };

  /// \class FileFormatException
  /// \brief Error handling for invalid file format
  class FileFormatException : public std::exception
  {
   public:
    /// \brief Constructor initializing exception
    /// \param errormessage Error message from input stream
    FileFormatException(const std::string_view errormessage) : std::exception(),
                                                               mMessage(std::string("Failure reading input file: ") + errormessage.data())
    {
    }

    /// \Destructor
    ~FileFormatException() noexcept override = default;

    /// \brief Access to error message of the exception
    /// \return Error message
    const char* what() const noexcept override
    {
      return mMessage.data();
    }

   private:
    std::string mMessage; ///< Error message from the input stream handling
  };

  /// \class InitStatusException
  /// \brief Error handling requests to not properly initialized mapping object
  class InitStatusException : public std::exception
  {
   public:
    /// \brief Constructor
    InitStatusException() = default;

    /// \brief Destructor
    ~InitStatusException() noexcept override = default;

    /// \brief Access to error message of the exception
    /// \return Error message
    const char* what() const noexcept override { return "Mapping not properly initialized"; }
  };

  /// \brief Default constructor
  Mapper() = default;

  /// \brief Costructor, initializing the mapping from file
  /// \param inputfile Name of the file containing the mapping
  /// \throw FileFormatException in case entries in the mapping file are not in the expected format
  /// \throw AddressRangeException in case addresses outside the valid range are found
  /// \throw ChannelTypeException in case hardware address with unkknown channel types are found
  Mapper(const std::string_view inputfile);

  /// \brief Destructor
  ~Mapper() = default;

  /// \brief Get channel ID params for a given hardware address
  /// \return Channel params corresponding to the hardware address
  /// \throw InitStatusException in case the mapping was not properly initialized
  /// \throw AddressNotFoundException in case of invalid hardware address
  ChannelID getChannelID(int hardawareaddress) const;

  /// \brief Get channel row for a given hardware address
  /// \return Row corresponding to the hardware address
  /// \throw InitStatusException in case the mapping was not properly initialized
  /// \throw AddressNotFoundException in case of invalid hardware address
  int getRow(int hardawareaddress) const
  {
    return getChannelID(hardawareaddress).mRow;
  }

  /// \brief Get channel column for a given hardware address
  /// \return Column corresponding to the hardware address
  /// \throw InitStatusException in case the mapping was not properly initialized
  /// \throw AddressNotFoundException in case of invalid hardware address
  int getColumn(int hardawareaddress) const
  {
    return getChannelID(hardawareaddress).mColumn;
  }

  /// \brief Get channel type for a given hardware address
  /// \return Channel type corresponding to the hardware address
  /// \throw InitStatusException in case the mapping was not properly initialized
  /// \throw AddressNotFoundException in case of invalid hardware address
  ChannelType_t getChannelType(int hardawareaddress) const
  {
    return getChannelID(hardawareaddress).mChannelType;
  }

  /// \brief Get the hardware address for a channel
  /// \param row Row of the channel
  /// \param col Column of the channel
  /// \param channeltype type of the channel
  /// \return Harware address of the channel
  /// \throw InitStatusException in case the mapping was not properly initialized
  int getHardwareAddress(int row, int col, ChannelType_t channeltype) const;

  /// \brief Initialize with new
  /// \param inputfile Name of the input file from which to read the mapping
  /// \throw FileFormatException in case entries in the mapping file are not in the expected format
  /// \throw AddressRangeException in case addresses outside the valid range are found
  /// \throw ChannelTypeException in case hardware address with unkknown channel types are found
  void setMapping(const std::string_view inputfile);

 private:
  /// \brief Costructor, initializing the mapping from file
  /// \param inputfile Name of the file containing the mapping
  /// \throw FileFormatException in case entries in the mapping file are not in the expected format
  /// \throw AddressRangeException in case addresses outside the valid range are found
  /// \throw ChannelTypeException in case hardware address with unkknown channel types are found
  void init(const std::string_view inputfile);

  std::unordered_map<int, ChannelID> mMapping;                         ///< Mapping between hardware address and col / row /caloflag
  std::unordered_map<ChannelID, int, ChannelIDHasher> mInverseMapping; ///< Inverse Mapping of channel type to hardware address
  bool mInitStatus = false;                                            ///< Initialization status

  ClassDefNV(Mapper, 1);
};

/// \brief stream operator for Mapper::Channel
/// \param stream Stream where the channel is displayed on
/// \param channel Channel to be displayed
/// \return Stream with channel
std::ostream& operator<<(std::ostream& stream, const Mapper::ChannelID& channel);

} // namespace emcal

} // namespace o2

#endif //__O2_EMCAL_MAPPER_H__
