// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/Serializer.h
/// \brief  Templated boost serializer/deserializer for MID vectors
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   02 December 2017
#ifndef O2_MID_SERIALIZER_H
#define O2_MID_SERIALIZER_H

#include <FairMQMessage.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace o2
{
namespace mid
{
/// Boost serializer for MID
template <typename DataType, typename BoostArchiveOut = boost::archive::binary_oarchive>
class Serializer
{
 public:
  void Serialize(FairMQMessage& msg, const std::vector<DataType>& dataVec)
  {
    /// Serislizes a standard vector of template type DataType into a message
    std::ostringstream buffer;
    BoostArchiveOut outputArchive(buffer);
    outputArchive << dataVec;
    int size = buffer.str().length();
    msg.Rebuild(size);
    std::memcpy(msg.GetData(), buffer.str().c_str(), size);
  }

  void Serialize(FairMQMessage& msg, const std::vector<DataType>& dataVec, const unsigned long nData)
  {
    /// Serializes a standard vector of template type DataType into a message
    /// when the number of elements to serialize is smaller tham the
    /// vector size
    std::vector<DataType> copyVec(dataVec.begin(), dataVec.begin() + nData);
    Serialize(msg, copyVec);
  }
};

/// Boost deserializer for MID
template <typename DataType, typename BoostArchiveIn = boost::archive::binary_iarchive>
class Deserializer
{
 public:
  void Deserialize(FairMQMessage& msg, std::vector<DataType>& input)
  {
    /// Deserializes the message into a standard vector of template type DataType
    input.clear();
    std::string msgStr(static_cast<char*>(msg.GetData()), msg.GetSize());
    std::istringstream buffer(msgStr);
    BoostArchiveIn inputArchive(buffer);
    inputArchive >> input;
  }
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_SERIALIZER_H */
