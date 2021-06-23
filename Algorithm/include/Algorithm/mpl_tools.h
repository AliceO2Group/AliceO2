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

//-*- Mode: C++ -*-

#ifndef MPL_TOOLS_H
#define MPL_TOOLS_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   mpl_tools.h
//  @author Matthias Richter
//  @since  2016-09-09
//  @brief  Tools for using the boost MPL

#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/bind.hpp>

using boost::mpl::placeholders::_;
using boost::mpl::placeholders::_1;
using boost::mpl::placeholders::_2;

namespace o2
{
namespace mpl
{

/**
 * @brief type trait to transparently use sequences of types
 * This type trait simply forwards to the given type if it is already a
 * sequence but wraps any non-sequence type into an mpl vector with one entry.
 *
 * Usage: make_mpl_vector<T>::type
 *        make_mpl_vector<T>::isSequence
 */
template <typename T>
class make_mpl_vector
{
 private:
  // the default implementaton of this trait expects the type T to be an
  // mpl sequence with a begin meta function.
  template <typename U, typename _Iter>
  struct VectorTraits {
    enum { result = true };
    using type = typename U::type;
  };
  // specialization for non sequences, for any data type which has no
  // begin meta function defined, the iterator is void type and this
  // specialization kicks in. The specialization wraps the type into
  // a sequence with one entry
  template <typename U>
  struct VectorTraits<U, boost::mpl::void_> {
    enum { result = false };
    using type = typename boost::mpl::fold<boost::mpl::vector<>,                 //
                                           boost::mpl::set<U>,                   //
                                           boost::mpl::insert<boost::mpl::_1, U> //
                                           >::type;                              //
  };

 public:
  /// iSequence tells if the original type is a sequence
  enum { isSequence = VectorTraits<T, typename boost::mpl::begin<T>::type>::result };
  /// the tarits type, always a sequence
  using type = typename VectorTraits<T, typename boost::mpl::begin<T>::type>::type;
};

/******************************************************************************
 * @brief Meta program to create a vector of wrapped types from a sequencs
 * of types. A new mpl vector containing Wrapper<Type> for every
 * element of the original sequence is created
 * mpl::fold recursivly applies the elements of the list to the previous result
 * placeholder _2 refers to the element, placeholder _1 to the previous result
 * or initial condition for the first element
 */
template <typename Seq, typename Wrapper>
struct do_typewrap {
  using type = typename boost::mpl::fold<                     //
    Seq,                                                      //
    boost::mpl::vector<>,                                     //
    boost::mpl::push_back<_1, boost::mpl::bind1<Wrapper, _2>> //
    >::type;
};

/******************************************************************************
 * @brief apply functor to element at mpl sequence position
 * This meta function recurses through the list while incrementing the index
 * and calls the functor at the required position
 *
 * @note this is a helper macro to the apply_at function
 */
template <typename _IndexT,   // data type of position index
          typename _Iterator, // current iterator position
          typename _End,      // end iterator position
          typename _ReturnT,  // type of the return value
          _IndexT _Index,     // current index
          typename F          // functor
          >
struct apply_at_sequence_position {
  static _ReturnT apply(_IndexT position, F f)
  {
    if (position == _Index) {
      // this is the queried position, execute function and
      // terminate loop by forwarding _End as _Iterator and thus
      // calling the specialization
      // TODO: check the performance penalty of the element instantiation
      typename boost::mpl::deref<_Iterator>::type element;
      return f(element);
    } else {
      // go to next element
      return apply_at_sequence_position<            //
        _IndexT,                                    //
        typename boost::mpl::next<_Iterator>::type, //
        _End,                                       //
        _ReturnT,                                   //
        _Index + 1,                                 //
        F                                           //
        >::apply(position, f);
    }
  }
};
// specialization: end of recursive loop, kicks in if _Iterator matches
// _End.
// here we end up if the position parameter is out of bounds
template <typename _IndexT, typename _End, typename _ReturnT, _IndexT _Index, typename F>
struct apply_at_sequence_position<_IndexT, _End, _End, _ReturnT, _Index, F> {
  static _ReturnT apply(_IndexT position, F f)
  {
    // TODO: this is probably the place to through an exeption because
    // we are out of bound
    return _ReturnT(0);
  }
};

/******************************************************************************
 * @brief A default functor with void return type and no operation
 */
struct defaultFunctor {
  using return_type = void;
  template <typename T>
  return_type operator()(T)
  {
  }
};

/******************************************************************************
 * @brief Apply a functor to an element of a compile time sequence
 * This meta function is a bridge to the runtime environment to apply a functor
 * to an element of a compile time list of types.
 * Required template parameter: sequence - typename defining the list of types
 *
 * @arg position   position of element in the list
 * @arg f          functor
 */
template <typename _Sequence, typename _IndexT = int, typename F = defaultFunctor>
typename F::return_type apply_at(_IndexT position, F f)
{
  return apply_at_sequence_position<_IndexT, typename boost::mpl::begin<_Sequence>::type,
                                    typename boost::mpl::end<_Sequence>::type, typename F::return_type, 0,
                                    F>::apply(position, f);
}

} // namespace mpl
} // namespace o2
#endif
