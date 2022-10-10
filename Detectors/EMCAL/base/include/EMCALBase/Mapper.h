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
#ifndef __O2_EMCAL_MAPPER_H__
#define __O2_EMCAL_MAPPER_H__

#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <iosfwd>
#include <unordered_map>
#include <sstream>
#include <string>
#include <boost/container_hash/hash.hpp>

#include "fmt/format.h"
#include "RStringView.h"
#include "Rtypes.h"
#include "DataFormatsEMCAL/Constants.h"

namespace o2
{

namespace emcal
{

/// \class Mapper
/// \brief ALTRO mapping for calorimeters
/// \ingroup EMCALbase
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
    ChannelType_t mChannelType; ///< Type of the channel (see o2::emcal::ChannelType for channel type definitions)

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
      std::size_t seed = 0;
      boost::hash_combine(seed, s.mRow);
      boost::hash_combine(seed, s.mColumn);
      boost::hash_combine(seed, o2::emcal::channelTypeToInt(s.mChannelType));
      return seed;
      /*
      size_t h1 = std::hash<int>()(s.mRow);
      size_t h2 = std::hash<int>()(s.mColumn);
      size_t h3 = std::hash<int>()(o2::emcal::channelTypeToInt(s.mChannelType));
      return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
      */
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
                                            mMessage()
    {
      std::stringstream msgbuilder;
      msgbuilder << "Hardware address " << address << "(0x" << std::hex << address << std::dec << ") not found";
      mMessage = msgbuilder.str();
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

    /// \brief Destructor
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
  ChannelID getChannelID(unsigned int hardawareaddress) const;

  /// \brief Get channel row for a given hardware address
  /// \return Row corresponding to the hardware address
  /// \throw InitStatusException in case the mapping was not properly initialized
  /// \throw AddressNotFoundException in case of invalid hardware address
  uint8_t getRow(unsigned int hardawareaddress) const
  {
    return getChannelID(hardawareaddress).mRow;
  }

  /// \brief Get channel column for a given hardware address
  /// \return Column corresponding to the hardware address
  /// \throw InitStatusException in case the mapping was not properly initialized
  /// \throw AddressNotFoundException in case of invalid hardware address
  uint8_t getColumn(unsigned int hardawareaddress) const
  {
    return getChannelID(hardawareaddress).mColumn;
  }

  /// \brief Get channel type for a given hardware address
  /// \return Channel type corresponding to the hardware address
  /// \throw InitStatusException in case the mapping was not properly initialized
  /// \throw AddressNotFoundException in case of invalid hardware address
  ChannelType_t getChannelType(unsigned int hardawareaddress) const
  {
    return getChannelID(hardawareaddress).mChannelType;
  }

  /// \brief Get the hardware address for a channel
  /// \param row Row of the channel
  /// \param col Column of the channel
  /// \param channeltype type of the channel
  /// \return Harware address of the channel
  /// \throw InitStatusException in case the mapping was not properly initialized
  unsigned int getHardwareAddress(uint8_t row, uint8_t col, ChannelType_t channeltype) const;

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

  std::unordered_map<unsigned int, ChannelID> mMapping;                         ///< Mapping between hardware address and col / row /caloflag
  std::unordered_map<ChannelID, unsigned int, ChannelIDHasher> mInverseMapping; ///< Inverse Mapping of channel type to hardware address
  bool mInitStatus = false;                                                     ///< Initialization status

  ClassDefNV(Mapper, 1);
};

/// \class MappingHandler
/// \brief Handler providing the correct mapping for the given DDL
///
/// EMCAL channel mapping consists of 4 mappings, organized in
/// - A- and C-side
/// - First or second DDL within the supermodule
/// The mapping handler provides user-friendly access to the correct
/// mapping for a given DDL automatically determining the proper mapping
/// data based on side and DDL in supermodule calculated from the DDL ID
class MappingHandler
{
 public:
  /// \class DDLInvalid
  /// \brief Error handling for invalid DDL IDs (not in range for EMCAL)
  ///
  /// Error thrown in queries to the MappingHandler where the DDL ID is
  /// out-of-range for EMCAL.
  class DDLInvalid final : public std::exception
  {
   public:
    DDLInvalid(int ddlID) : mDDL(ddlID) { mMessage = fmt::format("DDL {0} not existing for EMCAL", mDDL); };

    /// \brief Destructor
    ~DDLInvalid() noexcept final = default;

    /// \brief Access to the error message of the exception
    /// \return Error message
    const char* what() const noexcept final { return mMessage.data(); }

    /// \brief Access to DDL ID responsible for the exception
    /// \return DDL ID
    int getDDDL() const { return mDDL; }

   private:
    std::string mMessage; ///< error message
    int mDDL;             ///< DDL
  };

  /// \brief Constructor
  MappingHandler();

  /// \brief Destructor
  ~MappingHandler() = default;

  /// \brief Get Mapping for given DDL
  /// \param ddl ID of the DDL for which to get the mapping
  /// \return Mapping for the DDL (if valid)
  /// \throw DDLInvalid if DDL is invalid for EMCAL
  Mapper& getMappingForDDL(unsigned int ddl);

  /// \brief Get FEC index for channel based on DDL and information in the channel header
  /// \param ddl Absolute DDL index
  /// \param channelFEC FEC index in channel header
  /// \param branch Branch index (0 or 1) in DDL
  int getFEEForChannelInDDL(unsigned int dll, unsigned int channelFEC, unsigned int branch);

 private:
  std::array<Mapper, 4> mMappings; ///< Mapping container

  ClassDefNV(MappingHandler, 1);
};

/// \brief stream operator for Mapper::Channel
/// \param stream Stream where the channel is displayed on
/// \param channel Channel to be displayed
/// \return Stream with channel
std::ostream& operator<<(std::ostream& stream, const Mapper::ChannelID& channel);

} // namespace emcal

} // namespace o2

#endif //__O2_EMCAL_MAPPER_H__
