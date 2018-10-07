// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   BoostSerializer.h
/// \brief  Templatised boost serializer/deserializer for containers and base types
/// \author Gabriele G. Fronzé <gfronze at cern.ch>
/// \date   17 July 2018

#ifndef ALICEO2_BOOSTSERIALIZER_H
#define ALICEO2_BOOSTSERIALIZER_H

#include <utility>
#include <type_traits>
#include <array>
#include <vector>
#include <list>
#include <map>
#include <set>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/type_traits.hpp>

namespace o2
{
namespace utils
{
namespace check
{
// template <class Type, class Archive, typename = typename std::enable_if<std::is_class<Type>::value>::type>
template <typename...>
using void_t = void;

template <typename Type, typename Archive = boost::archive::binary_oarchive, typename = void_t<>>
struct is_boost_serializable_base : std::false_type {
};

template <class Type, typename Archive>
struct is_boost_serializable_base<Type, Archive,
                                  void_t<decltype(std::declval<Type &>().serialize(std::declval<Archive &>(), 0))>>
  : std::true_type {
};

// template <class Type, typename Archive>
// struct is_boost_serializable_base<Type, Archive,
//                                   typename std::enable_if<boost::serialization::is_bitwise_serializable<typename Type::value_type>::value>::type>
//   : std::true_type {
// };

template <class Type, typename Archive = boost::archive::binary_oarchive, typename = void_t<>>
struct is_boost_serializable
  : is_boost_serializable_base<Type, Archive> {
};

template <class Type, typename Archive>
struct is_boost_serializable<Type, Archive, void_t<typename Type::value_type>>
  : is_boost_serializable<typename Type::value_type, Archive> {
};

template <class Type>
struct is_boost_serializable<Type, boost::archive::binary_oarchive, void_t<typename Type::value_type>>
  : is_boost_serializable<typename Type::value_type, boost::archive::binary_oarchive> {
};
} // namespace check

template <typename ContT>
typename std::enable_if<check::is_boost_serializable<ContT, boost::archive::binary_oarchive>::value, std::ostringstream>::type
  BoostSerialize(const ContT& dataSet)
{
  /// Serialises a container (vector, array or list) using boost serialisation routines.
  /// Requires the contained type to be either trivial or provided with an overried of boost::serialise method.
  std::ostringstream buffer;
  boost::archive::binary_oarchive outputArchive(buffer);
  outputArchive << dataSet;
  return buffer;
}

template <typename ContT>
typename std::enable_if<check::is_boost_serializable<ContT, boost::archive::binary_iarchive>::value, ContT>::type
  BoostDeserialize(std::string& msgStr)
{
  /// Deserialises a msg contained in a string in a container type (vector, array or list) of the provided type.
  ContT output;
  std::istringstream buffer(msgStr);
  boost::archive::binary_iarchive inputArchive(buffer);
  inputArchive >> output;
  return std::move(output);
}
} // namespace utils
} // namespace o2

#endif //ALICEO2_BOOSTSERIALIZER_H
