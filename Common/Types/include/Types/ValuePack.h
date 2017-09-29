// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TYPES_VALUEPACK_H
#define TYPES_VALUEPACK_H
/// @file   ValuePack.h
/// @author Matthias Richter, Sandro Wenzel
/// @since  2017-09-28
/// @brief  A compact representation of bits from multiple values

#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/vector.hpp>
#include "boost/mpl/size.hpp"
#include <boost/mpl/pair.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/int.hpp>
#include "boost/mpl/at.hpp"
#include <boost/mpl/fold.hpp>
#include <boost/mpl/placeholders.hpp>

namespace mpl = boost::mpl;
using _1 = mpl::_1;
using _2 = mpl::_2;

namespace o2 {
namespace types {

//////////////////////////////////////////////////////////////////////////
/// Recurssively build the code
template<typename Rep, typename Iterator, typename End>
struct CodeInit {
  using Shift = typename boost::mpl::deref<Iterator>::type;
  static const Rep mask = (Rep(0x1) << Shift::value) - 1;

  template<typename... Types>
  static Rep value(size_t arg, Types... args) {
    return (arg & mask) | CodeInit<Rep, typename boost::mpl::next<Iterator>::type,
                                   End
                                   >::value(args...) << Shift::value;
  }
};

// specialization terminating the recursive meta function
template<typename Rep, typename End>
struct CodeInit<Rep, End, End> {
  static Rep value() {return 0;}
};

/**
* @class ValuePack
*
*/
template<typename Rep, size_t... Layout>
class ValuePack {
public:
  using self_type = ValuePack;
  using value_type = Rep;
  using Fields = typename boost::mpl::vector_c<size_t, Layout...>;
  static const size_t size = sizeof(value_type);
  static const size_t nbits = mpl::fold<
    Fields, mpl::int_<0>, mpl::plus<_1, _2>
    >::type::value;
  static const size_t nfields = boost::mpl::size<Fields>::type::value;
  static_assert(nbits <= size * 8, "representing type too narrow");

  /// default constructor all fields will be 0
  ValuePack() = default;

  /// constructor, the number of arguments must match the Layout
  /// template parameter
  template<typename... Types>
  ValuePack(Types... args) {
    static_assert(sizeof...(Types) == nfields, "incorrect number of arguments");
    valuepack = 0;
    valuepack |= CodeInit<
      value_type,
      typename boost::mpl::begin<Fields>::type,
      typename boost::mpl::end<Fields>::type
      >::value(args...);
  }

  ValuePack(const self_type& rhs)
    : valuepack(rhs.valuepack)
  {
    // TODO: could support a type which is effectively a subset of this type
  }

  decltype(auto) operator=(const self_type& rhs) {
    // TODO: could support a type which is effectively a subset of this type
    valuepack = rhs.valuepack;
    return *this;
  }

  bool operator==(const self_type& rhs) const {return valuepack == rhs.valuepack;}

  bool operator!=(const self_type& rhs) const {return valuepack != rhs.valuepack;}

  /// operator less than required for maps
  /// TODO: think about a customn function which can be provided as functor
  /// type or a lambda
  bool operator<(const self_type& rhs) const {return valuepack < rhs.valuepack;}

  /// type cast operator
  operator value_type() const {return valuepack;}

  /// set the field at position
  template<size_t Position, typename T>
  void set(T fieldvalue) {
    static_assert(Position < nfields, "index out of range");
    using FieldWidth = typename boost::mpl::at_c<Fields, Position>::type;
    static_assert(FieldWidth::value <= sizeof(T) * 8, "type too narrow to get value");
    using Shift = GetShift<Fields, Position>;
    static const auto mask = (value_type(0x1) << FieldWidth::value) - 1;

    valuepack &= ~(mask << Shift::value);
    valuepack |= (fieldvalue & mask) << Shift::value;
  }

  /// get the value of field at position
  template<size_t Position, typename T>
  T get() const {
    static_assert(Position < nfields, "index out of range");
    using FieldWidth = typename boost::mpl::at_c<Fields, Position>::type;
    static_assert(FieldWidth::value <= sizeof(T) * 8, "type too narrow to get value");
    using Shift = GetShift<Fields, Position>;
    static const auto mask = (value_type(0x1) << FieldWidth::value) - 1;

    return (valuepack >> Shift::value) & mask;
  }

private:
  /// calculate bit shift for field at position, i.e. sum of width of
  /// preceeding fields
  template<typename Sequence, size_t Position>
  struct GetShift {
    using type = typename mpl::fold<
      // the sequence to iterate over, elements provided in placeholder _2
      Sequence,
      // initial condition with a pair of two numbers, first one is increment
      // for every element and checked against position argument, second number
      // accumulates the sum until position is reached
      mpl::pair<mpl::int_<0>, mpl::int_<0>>,
      // the operation for every element of the sequence
      mpl::pair<mpl::next< mpl::first<_1>>,
              // if first is less than position, add element to second
              // forward second unchanged otherwise
                mpl::if_<mpl::less<mpl::first<_1>, mpl::int_<Position>>,
                         mpl::plus< mpl::second<_1>, _2 >,
                         mpl::second<_1>
                         >
                >
      >::type::second; // take the second number which accumulated the sum
    static const size_t value = type::value;
  };

  value_type valuepack = 0;
};


} // namespace types
} // namespace o2

#endif //TYPES_VALUEPACK_H
