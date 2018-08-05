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
///A set of classes and struct to be sure the serialised object is either trivial or implementing custom serialize
template <class Type, class Archive, typename = typename std::enable_if<std::is_class<Type>::value>::type>
class is_boost_serializable
{
 private:
  struct TypeOverloader {
    void serialize(Archive& ar, const unsigned int version) {}
  };
  struct TypeExt : public Type, public TypeOverloader {
  };
  template <typename T, T t>
  class DeductionHelper
  {
  };

  class True
  {
    char m;
  };
  class False
  {
    True m[2];
  };

  template <typename TestType>
  static False deduce(TestType*, DeductionHelper<void (TypeOverloader::*)(), &TestType::serialize>* = 0);
  static True deduce(...);

 public:
  static const bool value = (sizeof(True) == sizeof(deduce((TypeExt*)(0))));
};
} // namespace check

template <typename ContT>
typename std::enable_if<check::is_boost_serializable<ContT, boost::archive::binary_oarchive>::value
                        && std::is_class<typename ContT::value_type>::value, std::ostringstream>::type
  BoostSerialize(const ContT &dataSet)
{
  static_assert(check::is_boost_serializable<typename ContT::value_type, boost::archive::binary_oarchive>::value,
                "This class doesn't provide a boost serializer.");
  /// Serialises a container (vector, array or list) using boost serialisation routines.
  /// Requires the contained type to be either trivial or provided with an overried of boost::serialise method.
  std::ostringstream buffer;
  boost::archive::binary_oarchive outputArchive(buffer);
  outputArchive << dataSet;
  return buffer;
}

template <typename ContT, typename ContentT = typename ContT::value_type>
typename std::enable_if<check::is_boost_serializable<ContT, boost::archive::binary_oarchive>::value
                        && !(std::is_class<ContentT>::value), std::ostringstream>::type
  BoostSerialize(const ContT &dataSet)
{
  static_assert(boost::serialization::is_bitwise_serializable<typename ContT::value_type>::value,
                "This type doesn't provide a boost serializer.");
  /// Serialises a container (vector, array or list) using boost serialisation routines.
  /// Requires the contained type to be either trivial or provided with an overried of boost::serialise method.
  std::ostringstream buffer;
  boost::archive::binary_oarchive outputArchive(buffer);
  outputArchive << dataSet;
  return buffer;
}

template <typename ContT>
typename std::enable_if<check::is_boost_serializable<ContT, boost::archive::binary_iarchive>::value
                        && std::is_class<typename ContT::value_type>::value, ContT>::type
  BoostDeserialize(std::string &msgStr)
{
  static_assert(check::is_boost_serializable<typename ContT::value_type, boost::archive::binary_oarchive>::value,
                "This class doesn't provide a boost deserializer.");
  /// Deserialises a msg contained in a string in a container type (vector, array or list) of the provided type.
  ContT output;
  std::istringstream buffer(msgStr);
  boost::archive::binary_iarchive inputArchive(buffer);
  inputArchive >> output;
  return std::move(output);
}

template <typename ContT, typename ContentT = typename ContT::value_type>
typename std::enable_if<check::is_boost_serializable<ContT, boost::archive::binary_iarchive>::value
                        && !(std::is_class<ContentT>::value), ContT>::type
  BoostDeserialize(std::string &msgStr)
{
  static_assert(boost::serialization::is_bitwise_serializable<typename ContT::value_type>::value,
                "This type doesn't provide a boost serializer.");
  /// Deserialises a msg contained in a string in a container type (vector, array or list) of the provided type.
  ContT output;
  std::istringstream buffer(msgStr);
  boost::archive::binary_iarchive inputArchive(buffer);
  inputArchive >> output;
  return std::move(output);
}

template <typename T>
struct has_serializer
{
  template <class, class> class checker;

  template <typename C>
  static std::true_type test(checker<C, decltype(&o2::utils::BoostSerialize<C>)> *);

  template <typename C>
  static std::false_type test(...);

  typedef decltype(test<T>(nullptr)) type;
  static const bool value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};
} // namespace utils
} // namespace o2

#endif //ALICEO2_BOOSTSERIALIZER_H
