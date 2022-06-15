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

/* Local Variables:  */
/* mode: c++         */
/* End:              */

#ifndef DC_PRIMITIVES_H
#define DC_PRIMITIVES_H

/// @file   dc_primitives.h
/// @author Matthias Richter
/// @since  2016-08-15
/// @brief  Primitives for data compression

/**
 * The following template classes are defined
 * - ExampleAlphabet                example for alphabet definition
 * - ContiguousAlphabet             integer number alphabet range [min, max]
 * - ZeroBoundContiguousAlphabet    integer number alphabet range [0, max]
 * - ProbabilityModel               statistics for alphabet symbols
 */

#include <map>
//#define BOOST_MPL_LIMIT_STRING_SIZE 32
#include <boost/mpl/size.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/string.hpp>

namespace o2
{
namespace data_compression
{

//////////////////////////////////////////////////////////////////////////////
//
// General meta programming tools
//
// All meta programs are evaluated at compile time

/**
 * @brief Get maximum of an n-bit number
 *
 * Example usage: getmax<uint16_t, 13>
 */
template <typename T, std::size_t nbits>
struct getmax {
  static T const value = getmax<T, nbits - 1>::value << 1 | 1;
};
// template specialization for termination of the meta program
template <typename T>
struct getmax<T, 0> {
  static T const value = 0;
};

/**
 * @brief Get number of elements in a sequence of integral types
 * This redirects to the size meta program of the boost mpl, but includes the
 * upper bound as a valid element in the accounting.
 *
 * Example usage: getnofelements<int, -10, 10>::value
 */
template <typename T, T _min, T _max>
struct getnofelements {
  static std::size_t const value = boost::mpl::size<boost::mpl::range_c<T, _min, _max>>::value + 1;
};

/**
 * @brief Get the upper binary bound of a number
 * The gives the number of bits required to present a number
 *
 * Usage: upperbinarybound<number>::value
 */
template <std::size_t n>
struct upperbinarybound {
  static std::size_t const value = upperbinarybound<(n >> 1)>::value + 1;
};
// template specialization for termination of the meta program
template <>
struct upperbinarybound<0> {
  static std::size_t const value = 0;
};

//////////////////////////////////////////////////////////////////////////////
//
// DataCompression tools
//
// The Alphabet: for every parameter to be stored in a compressed format, there
// is a range of valid symbols referred to be an alphabet. The DataCompression
// framework requires an implementation of an alphabet which is then used with
// the probability model and codecs. The alphabet class needs to implement a
// check function (isValid), and a forward iterator together with begin and end
// function to hook it up to the framework.
//
// Default Alphabet implementations: most of the alphabets will describe a range
// of contiguous values, which is implemented by the class ContiguousAlphabet.
// Two specializations describe symbols in a range from 0 to a max value and
// within a certain bit range, respectively
// ZeroBoundContiguousAlphabet and BitRangeContiguousAlphabet

/******************************************************************************
 * @class ExampleAlphabet
 * @brief An example for an Alphabet definition.
 *
 * A class definition for an alphabet. Note that the functions are only defined,
 * not implemented. The class is required to support the following functions to
 * hook it up onto the DataCompression framework:
 * - isValid validity of a value
 * - getIndex  get index from value
 * - getSymbol get symbol from index
 * - forward iterator class to walk through elements of alphabet
 * - begin   start iteration over elements
 * - end     end marker for iteration
 *
 * The alphabet has to provide an index for the symbol range such that the
 * framework can build a one to one relation between symbols and index values
 * used for internal mapping of the symbols.
 */
template <typename T>
class ExampleAlphabet
{
 public:
  ExampleAlphabet();
  ~ExampleAlphabet();

  using value_type = T;

  /// check for valid value within range
  static bool isValid(value_type v);

  /// get index of value
  static unsigned getIndex(value_type symbol);

  /// get symbol from index
  static value_type getSymbol(unsigned index);

  /// get the range of indices aka number of indices
  constexpr unsigned getIndexRange();


  /// a forward iterator to access the list of elements
  template <typename ValueT>
  class Iterator
  {
   public:
    Iterator();
    ~Iterator();

    using self_type = Iterator;
    using value_type = ValueT;
    using reference = ValueT&;
    using pointer = ValueT*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    // prefix increment
    self_type& operator++();
    // postfix increment
    self_type operator++(int /*unused*/);
    // addition
    // self_type operator+(size_type n) const;
    // reference
    reference operator*();
    // comparison
    bool operator==(const self_type& other);
    // comparison
    bool operator!=(const self_type& other);

   private:
  };

  using iterator = Iterator<value_type>;
  using const_iterator = Iterator<const value_type>;

  /// return forward iterator to begin of element list
  const_iterator begin() const;
  /// the end of element list
  const_iterator end() const;
  /// return forward iterator to begin of element list
  iterator begin();
  /// the end of element list
  iterator end();

 private:
};

/******************************************************************************
 * Definition of a contiguous alphabet
 *
 * The contiguous alphabet defines integer number elements of the specified
 * type in a given range.
 *
 * TODO: the mpl string length is limited and can be configured with the
 *       define BOOST_MPL_LIMIT_STRING_SIZE (somehow the number of allowed
 *       template arguments is given by this number divided by four)
 */
template <typename T, T _min, T _max,                                                  //
          typename NameT = boost::mpl::string<'U', 'n', 'n', 'a', 'm', 'e', 'd'>::type //
          >
class ContiguousAlphabet
{
 public:
  ContiguousAlphabet() = default;
  ~ContiguousAlphabet() = default;

  using value_type = T;
  using size_type = T;
  using range = boost::mpl::range_c<T, _min, _max>;
  using size = boost::mpl::plus<boost::mpl::size<range>, boost::mpl::int_<1>>;

  /// check for valid value within range
  static bool isValid(value_type v) { return v >= _min && v <= _max; }

  /// get index of symbol
  ///
  /// Each alphabet has to provide a one to one mapping of symbols to
  /// index values used for internal storage
  /// For performance reasons, there is no range check
  static unsigned getIndex(value_type symbol)
  {
    int index = symbol;
    if (_min < 0) {
      index += -_min;
    } else if (_min > 0) {
      index -= _min;
    }
    return index;
  }

  /// get symbol from index
  static value_type getSymbol(unsigned index) { return _min + index; }

  /// get the range of indices aka number of indices
  constexpr unsigned getIndexRange() { return _max - _min; }

  /// get the name of the alphabet
  ///
  /// name is part of the type definition, defined as a boost mpl string
  constexpr const char* getName() const { return boost::mpl::c_str<NameT>::value; }

  template <typename ValueT>
  using _iterator_base = std::iterator<std::forward_iterator_tag, ValueT>;

  /// a forward iterator to access the list of elements
  template <typename ValueT>
  class Iterator : public _iterator_base<ValueT>
  {
   public:
    Iterator() : mValue(_max), mIsEnd(true) {}
    Iterator(T value, bool isEnd) : mValue(value), mIsEnd(isEnd) {}
    ~Iterator() = default;

    using self_type = Iterator;
    using value_type = typename _iterator_base<ValueT>::value_type;
    using reference = typename _iterator_base<ValueT>::reference;
    using pointer = typename _iterator_base<ValueT>::pointer;

    // prefix increment
    self_type& operator++()
    {
      if (mValue < _max) {
        mValue++;
      } else {
        mIsEnd = true;
      }
      return *this;
    }

    // postfix increment
    self_type operator++(int /*unused*/)
    {
      self_type copy(*this);
      ++*this;
      return copy;
    }

    // addition
    self_type operator+(size_type n) const
    {
      self_type copy(*this);
      if (!copy.mIsEnd) {
        if ((n > _max) || (_max - n < mValue)) {
          copy.mIsEnd = true;
          copy.mValue = _max;
        } else {
          copy.mValue += n;
        }
      }
      return copy;
    }

    reference operator*() { return mValue; }
    // pointer operator->() const {return &mValue;}
    // reference operator[](size_type n) const;

    bool operator==(const self_type& other) { return mValue == other.mValue && mIsEnd == other.mIsEnd; }
    bool operator!=(const self_type& other) { return not(*this == other); }

   private:
    value_type mValue;
    bool mIsEnd;
  };

  using iterator = Iterator<value_type>;
  using const_iterator = Iterator<const value_type>;

  /// return forward iterator to begin of element list
  const_iterator begin() const { return iterator(_min, false); }

  /// the end of element list
  const_iterator end() const { return iterator(_max, true); }

  /// return forward iterator to begin of element list
  iterator begin() { return iterator(_min, false); }

  /// the end of element list
  iterator end() { return iterator(_max, true); }

 private:
};

/******************************************************************************
 * Definition of a zero-bound contiguous alphabet
 *
 * The zero-bound contiguous alphabet defines integer number elements of the
 * specified type between 0 and a maximum value.
 */
template <typename T, T _max,                                                          //
          typename NameT = boost::mpl::string<'U', 'n', 'n', 'a', 'm', 'e', 'd'>::type //
          >
class ZeroBoundContiguousAlphabet : public ContiguousAlphabet<T, 0, _max, NameT>
{
};

/******************************************************************************
 * Definition of a bit-range contiguous alphabet
 *
 * The bit-range contiguous alphabet defines integer number elements between
 * 0 and the maximum number allowed by bit-range
 */
template <typename T, std::size_t _nbits, typename NameT = boost::mpl::string<'U', 'n', 'n', 'a', 'm', 'e', 'd'>::type>
class BitRangeContiguousAlphabet : public ZeroBoundContiguousAlphabet<T, getmax<T, _nbits>::value, NameT>
{
};

/******************************************************************************
 * Probability model class collecting statistics for an alphabet
 *
 */
template <class Alphabet, typename WeightType = double>
class ProbabilityModel
{
 public:
  using alphabet_type = Alphabet;
  using value_type = typename Alphabet::value_type;
  using weight_type = WeightType;
  using TableType = std::map<value_type, WeightType>;

  // TODO: check if this is the correct way for WeightType defaults
  static const value_type _default0 = 0;
  static const value_type _default1 = _default0 + 1;

  ProbabilityModel() : mProbabilityTable(), mTotalWeight(_default0) {}
  ~ProbabilityModel() = default;

  constexpr const char* getName() const
  {
    Alphabet tmp;
    return tmp.getName();
  }

  int addWeight(value_type value, weight_type weight = _default1)
  {
    mProbabilityTable[value] += weight;
    mTotalWeight += weight;
    return 0;
  }

  int initWeight(Alphabet& alphabet, WeightType weight = _default1)
  {
    mProbabilityTable.clear();
    mTotalWeight = _default0;
    for (auto i : alphabet) {
      addWeight(i, weight);
    }
    return 0;
  }

  WeightType normalize()
  {
    WeightType totalWeight = _default0;
    // TODO: handle division by zero, although that should not occur at all
    for (typename TableType::iterator i = mProbabilityTable.begin(); i != mProbabilityTable.end(); i++) {
      totalWeight += i->second;
      i->second /= mTotalWeight;
    }
    // TODO: verify total weight
    // if (mTotalWeight - verifyTotalWeight > some small value)
    mTotalWeight = totalWeight / mTotalWeight;
    return totalWeight;
  }

  // const reference only to avoid changes in the weight count
  // without registering in the total weight as well
  const WeightType& operator[](value_type v) const
  {
    typename TableType::const_iterator i = mProbabilityTable.find(v);
    if (i != mProbabilityTable.end()) {
      return i->second;
    }
    static WeightType dummy = _default0;
    return dummy;
  }

  typename TableType::const_iterator begin() const { return mProbabilityTable.begin(); }

  typename TableType::const_iterator end() const { return mProbabilityTable.end(); }

  typename TableType::iterator begin() { return mProbabilityTable.begin(); }

  typename TableType::iterator end() { return mProbabilityTable.end(); }

  void print() const {}

 private:
  TableType mProbabilityTable;
  WeightType mTotalWeight;
};

} // namespace data_compression
} // namespace o2
#endif
