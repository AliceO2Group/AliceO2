//-*- Mode: C++ -*-

#ifndef RUNTIME_CONTAINER_H
#define RUNTIME_CONTAINER_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Author(s): Matthias Richter <mail@matthias-richter.com>          *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

/// @file   runtime_container.h
/// @author Matthias Richter
/// @since  2016-09-11
/// @brief  A general runtime container for a compile time sequence
/// This file is part of https://github.com/matthiasrichter/gNeric

// A general runtime container for a compile time sequence
// of types. A mixin class is used to represent a member of each data
// type. Every data type in the sequence describes a mixin on top of
// the previous one. The runtime container accumulates the type
// properties.

#include <boost/mpl/at.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/end.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/less.hpp>
#include <boost/mpl/minus.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/protect.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <iomanip>
#include <iostream>

using namespace boost::mpl::placeholders;

namespace gNeric
{
/**
 * @class DefaultInterface
 * @brief The default interface for the RuntimeContainer
 *
 * The common interface for the mixin class. In order to allow entry
 * points to the different levels of the mixin, none of the interface
 * functions has to be declared virtual. The function implementation of
 * the top most mixin would be called otherwise.
 *
 * The mixin technique requires a base class, but it mostly makes sense in
 * the picture of runtime polymorphism and virtual interfaces. The runtime
 * container application is purely using static polymorphism which makes the
 * base interface just to a technical aspect.
 */
class DefaultInterface
{
 public:
  DefaultInterface() {}
  ~DefaultInterface() {}
  void print() const {}
};

/**
 * @brief Default initializer does nothing
 */
struct default_initializer {
  template <typename T>
  void operator()(T&)
  {
  }
};

/**
 * @brief An initializer for simple types
 * The initializer makes use of truncation for non-float types, and
 * over- and underflow to produce different values in the member
 * of the individual stages in the container.
 * - float types keep the fraction
 * - integral types truncate the fraction
 * - unsigned types undergo an underflow and produce big numbers
 * - 8 bit char produces the '*' character
 *
 * Mainly for testing and illustration purposes.
 */
struct funny_initializer {
  template <typename T>
  void operator()(T& v)
  {
    v = 0;
    v -= 214.5;
  }
};

/**
 * @brief Default printer prints nothing
 */
struct default_printer {
  template <typename T>
  bool operator()(const T& v, int level = -1)
  {
    return false;
  }
};

/**
 * @brief Verbose printer prints level and content
 */
template <bool recursive = true>
struct verbose_printer_base {
  template <typename T>
  bool operator()(const T& v, int level = -1)
  {
    std::cout << "RC mixin level " << std::setw(2) << level << ": " << v << std::endl;
    return recursive;
  }
};

/**
 * @brief Verbose printer to print levels recursively
 */
struct recursive_printer : verbose_printer_base<true> {
};

// preserve backward compatibility
typedef recursive_printer verbose_printer;

/**
 * @brief Verbose printer to print a single level
 */
struct single_printer : verbose_printer_base<false> {
};

/**
 * @brief Setter functor, forwards to the container mixin's set function
 */
template <typename U>
class set_value
{
 public:
  typedef void return_type;
  typedef U value_type;

  set_value(U u) : mValue(u) {}
  template <typename T>
  return_type operator()(T& t)
  {
    *t = mValue;
  }

 private:
  set_value(); // forbidden
  U mValue;
};

/**
 * @brief Adder functor
 */
template <typename U>
class add_value
{
 public:
  typedef void return_type;
  typedef U value_type;

  add_value(U u) : mValue(u) {}
  template <typename T>
  return_type operator()(T& t)
  {
    *t += mValue;
  }

 private:
  add_value(); // forbidden
  U mValue;
};

/**
 * @brief Getter functor, forwards to the container mixin's get function
 *
 * TODO: make a type trait to either return t.get() if its a container
 * instance or t directly if it is the member object
 */
template <typename U>
class get_value
{
 public:
  typedef U return_type;
  typedef U value_type;
  class NullType
  {
  };

 private:
  /* could not solve the problem that one has to instantiate Traits
     with a fixed number of template arguments where wrapped_type
     would need to be provided already to go into the specialization
  template<typename InstanceType, typename Dummy = InstanceType>
  struct Traits {
    typedef NullType container_type;
    typedef InstanceType type;
    static return_type apply(InstanceType& c) {
      std::cout << "Traits";
      return c;
    }
  };
  // specialization for container instances
  template<typename InstanceType>
  struct Traits<InstanceType, typename InstanceType::wrapped_type> {
    typedef InstanceType container_type;
    typedef typename InstanceType::wrapped_type type;
    static return_type apply(InstanceType& c) {
      std::cout << "specialized Traits";
      return c.get();
    }
  };
  */

 public:
  template <typename T>
  return_type operator()(T& t)
  {
    return t.get();
    // return (typename Traits<T>::type)(t);
  }
};

/******************************************************************************
 * @brief apply functor to the wrapped member object in the runtime container
 * This meta function recurses through the list while incrementing the index
 * and calls the functor at the required position
 *
 * @note internal meta function for the RuntimeContainers' apply function
 */
template <typename _ContainerT // container type
          ,
          typename _IndexT // data type of position index
          ,
          typename _Iterator // current iterator position
          ,
          typename _End // end iterator position
          ,
          _IndexT _Index // current index
          ,
          typename F // functor
          >
struct rc_apply_at {
  static typename F::return_type apply(_ContainerT& c, _IndexT position, F& f)
  {
    if (position == _Index) {
      // this is the queried position, make the type cast to the current
      // stage of the runtime container and execute function for it.
      // Terminate loop by forwarding _End as _Iterator and thus
      // calling the specialization
      typedef typename boost::mpl::deref<_Iterator>::type stagetype;
      stagetype& stage = static_cast<stagetype&>(c);
      return f(stage);
    } else {
      // go to next element
      return rc_apply_at<_ContainerT, _IndexT, typename boost::mpl::next<_Iterator>::type, _End, _Index + 1, F>::apply(
        c, position, f);
    }
  }
};
// specialization: end of recursive loop, kicks in if _Iterator matches
// _End.
// here we end up if the position parameter is out of bounds
template <typename _ContainerT // container type
          ,
          typename _IndexT // data type of position index
          ,
          typename _End // end iterator position
          ,
          _IndexT _Index // current index
          ,
          typename F // functor
          >
struct rc_apply_at<_ContainerT, _IndexT, _End, _End, _Index, F> {
  static typename F::return_type apply(_ContainerT& c, _IndexT position, F& f)
  {
    // TODO: this is probably the place to throw an exeption because
    // we are out of bound
    return typename F::return_type(0);
  }
};

/**
 * Apply functor to the specified container level
 *
 * Ignores parameter '_IndexT'
 */
template <typename _ContainerT, typename _StageT, typename _IndexT, typename F>
struct rc_apply {
  typedef typename _ContainerT::types types;
  static typename F::return_type apply(_ContainerT& c, _IndexT /*ignored*/, F& f)
  {
    return f(static_cast<_StageT&>(c));
  }
};

/**
 * Generalized dispatcher with the ability for code unrolling
 *
 * The optional template parameter 'Position' can be used to cast directly to
 * the specified level in the runtime container and apply the functor without
 * the recursive loop. The template call with default parameters forwards to
 * the recursive call because 'Position' is set to out of list range.
 */
template <typename _ContainerT, typename F, typename Position = boost::mpl::size<typename _ContainerT::types>,
          typename _IndexT = int>
struct rc_dispatcher {
  typedef typename _ContainerT::types types;
  typedef typename boost::mpl::if_<boost::mpl::less<Position, boost::mpl::size<types>>,
                                   rc_apply<_ContainerT, typename boost::mpl::at<types, Position>::type, _IndexT, F>,
                                   rc_apply_at<_ContainerT, _IndexT, typename boost::mpl::begin<types>::type,
                                               typename boost::mpl::end<types>::type, 0, F>>::type type;

  static typename F::return_type apply(_ContainerT& c, _IndexT position, F& f) { return type::apply(c, position, f); }
};

/**
 * @class RuntimeContainer The base for the mixin class
 * @brief the technical base of the mixin class
 *
 * The class is necessary to provide the innermost functionality of the
 * mixin.
 *
 * The level of the mixin is encoded in the type 'level' which is
 * incremented in each mixin stage.
 */
template <typename InterfacePolicy = DefaultInterface, typename InitializerPolicy = default_initializer,
          typename PrinterPolicy = default_printer>
struct RuntimeContainer : public InterfacePolicy {
  InitializerPolicy _initializer;
  PrinterPolicy _printer;
  typedef boost::mpl::int_<-1> level;
  typedef boost::mpl::vector<>::type types;

  /// get size which is 0 at this level
  constexpr std::size_t size() const { return 0; }
  void print()
  {
    const char* string = "base";
    _printer(string, level::value);
  }

  // not yet clear if we need the setter and getter in the base class
  // at least wrapped_type is not defined in the base
  // void set(wrapped_type) {mMember = v;}
  // wrapped_type get() const {return mMember;}
};

/**
 * @class rc_mixin Components for the mixin class
 * @brief Mixin component is used with different data types
 *
 * Each mixin component has a member of the specified type. The container
 * level exports the following data types to the outside:
 * - wrapped_type    the data type at this level
 * - mixin_type      composed type at this level
 * - types           mpl sequence containing all level types
 * - level           a data type containing the level
 */
template <typename BASE, typename T>
class rc_mixin : public BASE
{
 public:
  rc_mixin() : mMember() { BASE::_initializer(mMember); }
  /// each stage of the mixin class wraps one type
  typedef T wrapped_type;
  /// this is the self type
  typedef rc_mixin<BASE, wrapped_type> mixin_type;
  /// a vector of all mixin stage types so far
  typedef typename boost::mpl::push_back<typename BASE::types, mixin_type>::type types;
  /// increment the level counter
  typedef typename boost::mpl::plus<typename BASE::level, boost::mpl::int_<1>>::type level;
  void print()
  {
    // use the printer policy of this level, the policy returns
    // a bool determining whether to call the underlying level
    if (BASE::_printer(mMember, level::value)) {
      BASE::print();
    }
  }

  /// get size at this stage
  constexpr std::size_t size() const { return level::value + 1; }
  /// set member wrapped object
  void set(wrapped_type v) { mMember = v; }
  /// get wrapped object
  wrapped_type get() const { return mMember; }
  /// get wrapped object reference
  wrapped_type& operator*() { return mMember; }
  /// assignment operator to wrapped type
  wrapped_type& operator=(const wrapped_type& v)
  {
    mMember = v;
    return mMember;
  }
  /// type conversion to wrapped type
  operator wrapped_type() const { return mMember; }
  /// operator
  wrapped_type& operator+=(const wrapped_type& v)
  {
    mMember += v;
    return mMember;
  }
  /// operator
  wrapped_type operator+(const wrapped_type& v) { return mMember + v; }
  /// a functor wrapper dereferencing the RC container instance
  /// the idea is to use this extra wrapper to apply the functor directly to
  /// the wrapped type, see the comment below
  template <typename F>
  class member_apply_at
  {
   public:
    member_apply_at(F& f) : mFunctor(f) {}
    typedef typename F::return_type return_type;
    template <typename _T>
    typename F::return_type operator()(_T& me)
    {
      return mFunctor(*me);
    }

   private:
    member_apply_at(); // forbidden
    F& mFunctor;
  };

  /// apply functor to the runtime object at index
  /// TODO: there is a performance issue with this solution, introducing another
  /// level of functors makes the access much slower compared with applying to
  /// container instance and using container member functions, tested with the
  /// add_value functor and bench_runtime_container, also the actual operation
  /// needs to be checked, the result is not correct for the last check of
  /// 100000000 iterations
  /*
  template<typename F>
  typename F::return_type applyToMember(int index, F f) {
    return apply(index, member_apply_at<F>(f));
  }
  */

  /*
   * Apply a functor to the runtime container at index
   *
   * For performance tests there is a template option to do an explicite loop
   * unrolling for the first n (=10) elements. This is however only effective
   * if the compiler optimization is switched of. This is  in the end a nice
   * demonstrator for the potential of compiler optimization. Unrolling is
   * switched on with the compile time switch RC_UNROLL.
   */
  template <typename F
#ifdef RC_UNROLL
            ,
            bool unroll = true
#else
            ,
            bool unroll = false
#endif
            >
  typename F::return_type apply(int index, F f)
  {
    if (unroll) { // this is a compile time switch
      // do unrolling for the first n elements and forward to generic
      // recursive function for the rest.
      switch (index) {
        case 0:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<0>, int>::apply(*this, 0, f);
        case 1:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<1>, int>::apply(*this, 1, f);
        case 2:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<2>, int>::apply(*this, 2, f);
        case 3:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<3>, int>::apply(*this, 3, f);
        case 4:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<4>, int>::apply(*this, 4, f);
        case 5:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<5>, int>::apply(*this, 5, f);
        case 6:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<6>, int>::apply(*this, 6, f);
        case 7:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<7>, int>::apply(*this, 7, f);
        case 8:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<8>, int>::apply(*this, 8, f);
        case 9:
          return rc_dispatcher<mixin_type, F, boost::mpl::int_<9>, int>::apply(*this, 9, f);
      }
    }
    return rc_dispatcher<mixin_type, F>::apply(*this, index, f);
  }

 private:
  T mMember;
};

/**
 * @brief Applying rc_mixin with the template parameters as placeholders
 * The wrapping into an mpl lambda is necessary to separate placeholder scopes
 * in the mpl fold operation.
 */
typedef typename boost::mpl::lambda<rc_mixin<_1, _2>>::type apply_rc_mixin;

/**
 * @brief check the mixin level to be below specified level
 *
 * @note: the number is specified as a type, e.g. boost::mpl:int_<3>
 */
template <typename T, typename N>
struct rtc_less : boost::mpl::bool_<(T::level::value < boost::mpl::minus<N, boost::mpl::int_<1>>::value)> {
};

template <typename T, typename N>
struct rtc_equal : boost::mpl::bool_<boost::mpl::equal<typename T::wrapped_type, N>::type> {
};

/**
 * @brief create the runtime container type
 * The runtime container type is build from a list of data types, the recursive
 * build can be optionally stopped at the level of argument N.
 *
 * Usage: typedef create_rtc<types, base>::type container_type;
 */
template <typename Types, typename Base, typename N = boost::mpl::size<Types>>
struct create_rtc {
  typedef typename boost::mpl::lambda<
    // mpl fold loops over all elements in the list of the first template
    // parameter and provides this as placeholder _2; for every element the
    // operation of the third template parameter is applied to the result of
    // the previous stage which is provided as placeholder _1 to the operation
    // and initialized to the second template argument for the very first
    // operation
    typename boost::mpl::fold<
      // list of types, each element provided as placeholder _1
      Types
      // initializer for the _1 placeholder
      ,
      Base
      // recursively applied operation, depending on the outcome of rtc_less
      // either the next mixin level is applied or the current state is used
      ,
      boost::mpl::if_<rtc_less<_1, N>
                      // apply mixin level
                      ,
                      boost::mpl::apply2<boost::mpl::protect<apply_rc_mixin>::type, _1, _2>
                      // keep current state by identity
                      ,
                      boost::mpl::identity<_1>>>::type>::type type;
};

/**
 * @brief create an mpl vector of mixin types
 * Every stage in the runtime container contains all the previous ones.
 * The resulting mpl vector of this meta function contains all individual
 * stages.
 *
 * Usage: typedef create_rtc_types<types, base>::type container_types;
 */
template <typename Types, typename Base, typename N = boost::mpl::size<Types>>
struct create_rtc_types {
  typedef typename boost::mpl::fold<
    boost::mpl::range_c<int, 0, N::value>, boost::mpl::vector<>,
    boost::mpl::push_back<_1, create_rtc<Types, Base, boost::mpl::plus<_2, boost::mpl::int_<1>>>>>::type type;
};

}; // namespace gNeric

#endif
