// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  @file   test_mpl_tools.cxx
//  @author Matthias Richter
//  @since  2017-06-22
//  @brief  Test program for MPL tools

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "../include/Algorithm/mpl_tools.h"
#include <boost/mpl/size.hpp>
#include <boost/type.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/string.hpp> // see note below
#include <iostream>
#include <iomanip>
#include <vector>
#include <type_traits>

// FIXME: mpl/string.hpp required to be included to avoid compilation error
// error: no matching function for call to ‘assertion_failed ...' in the mpl::for_each
// implementation (BOOST_MPL_ASSERT check). Not fully understood as simply including
// mpl/assert.hpp does not help. Maybe it's an inconssitency in the do_typewrap meta
// program
namespace bmpl = boost::mpl;

// defining a list of known data types
using knowntypes = bmpl::vector<float, double, long double, short, long>;

// get the index of an element in a type sequence
template <typename Iterator, typename End, typename Element, typename T, int Count = 0>
struct type_index {
  using type = typename bmpl::if_<typename std::is_same<Element, T>::type, bmpl::int_<Count>,   //
                                  typename type_index<typename bmpl::next<Iterator>::type, End, //
                                                      typename bmpl::deref<Iterator>::type,     //
                                                      T,                                        //
                                                      Count + 1                                 //
                                                      >::type>::type;                           //
  static const int value = type::value;
};
// termination condition
template <typename End, typename Element, typename T, int Count>
struct type_index<End, End, Element, T, Count> {
  using type = typename bmpl::if_<typename std::is_same<Element, T>::type, //
                                  bmpl::int_<Count>,                       //
                                  bmpl::int_<Count + 1>>::type;            //

  static const int value = type::value;
};

// Initializer for the inner type wrapper, use the index in the
// list of known types
template <typename T>
struct Initializer {
  static const int value = type_index<typename bmpl::begin<knowntypes>::type, //
                                      typename bmpl::end<knowntypes>::type,   //
                                      bmpl::void_,                            //
                                      T>::value;                              //
};

// the inner class for the fold
// the data member is initialized with the index in the list
// of known data types which can be wrapped by the class
template <typename T>
class Model
{
 public:
  using value_type = T;

  Model() : mData(Initializer<value_type>::value) {}
  ~Model() = default;

  friend std::ostream& operator<<(std::ostream& stream, const Model& rhs)
  {
    stream << "Model ";
    stream << rhs.mData;
    return stream;
  }

 private:
  T mData;
};

// the outer class for the fold
template <typename T>
class Codec
{
 public:
  using value_type = T;
  using wrapped_type = typename T::value_type;

  Codec() = default;
  ~Codec() = default;

  friend std::ostream& operator<<(std::ostream& stream, const Codec& rhs)
  {
    stream << "Codec ";
    stream << rhs.mData;
    return stream;
  }

 private:
  T mData;
};

// simple functor for printing class/type info
struct checktype {
  template <typename T>
  void operator()(const T& v)
  {
    std::cout << v << std::endl;
    std::cout << "is integral: " << std::is_integral<T>::value << std::endl;
  }
};

BOOST_AUTO_TEST_CASE(test_mpl_fold)
{
  using types = bmpl::vector<long, float, short, double, float, long, long double>;
  std::cout << std::endl
            << "checking types:" << std::endl;
  bmpl::for_each<types>(checktype());

  // bmpl::fold recursivly applies the elements of the list to the previous result
  // placeholder _2 refers to the element, placeholder _1 to the previous result
  // or initial condition for the first element
  // in this particular example, the initial condition is a data type for in 0,
  // which is incremented with bmpl::next if the type of the element is float
  using number_of_floats =
    bmpl::fold<types, bmpl::int_<0>, bmpl::if_<std::is_floating_point<_2>, bmpl::next<_1>, _1>>::type;
  // using the meta program, this can be a static_assert
  static_assert(number_of_floats::value == 4, "inconsistent number of float values in the type definition");

  // wrapping all elements into the Model class
  std::cout << std::endl
            << "checking first fold:" << std::endl;
  using models = o2::mpl::do_typewrap<types, bmpl::lambda<Model<_>>::type>::type;
  bmpl::for_each<models>(checktype());

  // wrapping all elements into the Codec class
  std::cout << std::endl
            << "checking second fold:" << std::endl;
  using codecs = o2::mpl::do_typewrap<models, bmpl::lambda<Codec<_>>::type>::type;
  bmpl::for_each<codecs>(checktype());
}
