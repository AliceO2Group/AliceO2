//-*- Mode: C++ -*-

#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H
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

//  @file   DataGenerator.h
//  @author Matthias Richter
//  @since  2016-12-06
//  @brief  A simple data generator

#include <stdexcept>  // exeptions, range_error
#include <utility>    // std::forward
#include <random>     // random distribution
#include <cmath>      // exp
//#include <iostream> // lets see if needed, or keep class fre from output
//#include <iomanip>  // lets see if needed, or keep class fre from output

namespace AliceO2 {
namespace Test {

/**
 * @class DataGenerator
 * @brief A simple data generator
 *
 * Generate random numbers according to the distribution model ModelT, which
 * also has to provide the formula. Some distribution models like e.g. normal
 * distribution work on float type values. The random numbers are ordered in
 * bins between min and max with the configured step width.
 *
 * The underlying distribution model must provide an analytic function to
 * calculate the probability for every bin.
 *
 * TODO:
 * - float numbers can not serve as tempate parameters, but maybe there
 *   is another way to move some of the computation to compile time
 * - number of bins: define policy for the last bin,
 *   what to do if (max-min)/step is not integral?
 * - consider returning the bin number instead of random number
 * - configurable seed
 * - error policy
 */
template<typename ValueT
         , typename ModelT>
class DataGenerator {
public:
  typedef int size_type;
  typedef ValueT result_type;
  typedef DataGenerator self_type;

  template<typename... Args>
  DataGenerator(result_type _min,
                result_type _max,
                result_type _step,
                Args&&... args)
    : mGenerator(), min(_min), max(_max), step(_step), nbins((max-min)/step), mModel(std::forward<Args>(args)...) {}
  ~DataGenerator() {}
  DataGenerator(const DataGenerator&) = default;
  DataGenerator& operator=(const DataGenerator&) = default;

  const result_type min;
  const result_type max;
  const result_type step;
  const size_type nbins;

  typedef ValueT value_type;
  typedef std::default_random_engine random_engine;

  /// get next random value
  // TODO: can it be const?
  value_type operator()() {
    value_type v;
    int trials = 0;
    while ((v = mModel(mGenerator)) < min || v >= max) {
      if (trials++ > 1000) {
        // this is a protection, just picked a reasonable threshold for number of trials
        throw std::range_error("random value outside configured range for too many trials");
      }
    }
    int bin = (v - min)/step;
    return min + bin * step;
  }

  /// get next random value
  value_type getRandom() const {return (*this)();}

  /// get minimum value
  value_type getMin() const {return ModelT::min;}

  /// get maximum value
  value_type getMax() const {return ModelT::max;}

  /// get theoretical probability of a value
  double getProbability(value_type v) const {
    return mModel.getProbability(v);
  }

  typedef std::iterator<std::forward_iterator_tag, result_type> _iterator_base;

  /**
   * @class iterator a forward iterator to access the bins
   *
   * TODO:
   * - check overhead by the computations in the deref operator
   */
  template<class ContainerT>
  class iterator : public _iterator_base {
  public:
    iterator(const ContainerT& parent, size_type count = 0) : mParent(parent), mCount(count) {}
    ~iterator() {}

    typedef iterator self_type;
    typedef typename _iterator_base::value_type value_type;
    typedef typename _iterator_base::reference reference;

    // prefix increment
    self_type& operator++() {
      if (mCount < mParent.nbins) mCount++;
      return *this;
    }

    // postfix increment
    self_type operator++(int /*unused*/) {self_type copy(*this); ++*this; return copy;}

    // addition
    self_type operator+(size_type n) const {
      self_type copy(*this);
      if (copy.mCount + n < mParent.nbins) {
        copy.mCount += n;
      } else {
        copy.mCount = mParent.nbins;
      }
      return copy;
    }

    value_type operator*() {return mParent.min + (mCount +.5) * mParent.step;}
    //pointer operator->() const {return &mValue;}
    //reference operator[](size_type n) const;

    bool operator==(const self_type& other) {
      return mCount == other.mCount;
    }
    bool operator!=(const self_type& other) {
      return not (*this == other);
    }

  private:
    const ContainerT& mParent;
    size_type mCount;
  };

  /// return forward iterator to begin of bins
  iterator<self_type> begin() {
    return iterator<self_type>(*this);
  }

  /// return forward iterator to the end of bins
  iterator<self_type> end() {
    return iterator<self_type>(*this, nbins);
  }

 private:
  random_engine mGenerator;
  ModelT mModel;
};

/**
 * @class normal_distribution
 * @brief specialization of std::normal_distribution which implements
 * also the analytic formula.
 */
template <class RealType = double
          , class _BASE = std::normal_distribution<RealType>
          >
class normal_distribution : public _BASE {
public:
  typedef typename _BASE::result_type result_type;

  normal_distribution(result_type _mean,
                      result_type _stddev
                      ) : _BASE(_mean, _stddev), mean(_mean), stddev(_stddev) {}

  const double sqrt2pi = 2.5066283;
  const result_type mean;
  const result_type stddev;

  /// get theoretical probability of a value
  // if value_type is an integral type we want to have the probability
  // that the result value is in the range [v, v+1) whereas the step
  // can be something else than 1
  // also the values outside the specified range should be excluded
  // and the probability for intervals in the range has to be scaled
  template<typename value_type>
  double getProbability(value_type v) const {
    return (exp(-(v-mean)*(v-mean)/(2*stddev*stddev)))/(stddev * sqrt2pi);
  }
};

/**
 * @class poisson_distribution
 * @brief specialization of std::poisson_distribution which implements
 * also the analytic formula.
 */
template <class IntType = int
          , class _BASE = std::poisson_distribution<IntType>
          >
class poisson_distribution : public _BASE {
public:
  typedef typename _BASE::result_type result_type;

  poisson_distribution(result_type _mean) : _BASE(_mean), mean(_mean) {}
  ~poisson_distribution() {};

  const result_type mean;

  int factorial(unsigned int n) const {
    return (n <= 1)? 1 : factorial(n-1) * n;
  }

  /// get theoretical probability of a value
  template<typename value_type>
  double getProbability(value_type v) const {
    if (v<0) return 0.;
    return pow(mean, v) * exp(-mean) / factorial(v);
  }
};

/**
 * @class geometric_distribution
 * @brief specialization of std::geometric_distribution which implements
 * also the analytic formula.
 */
template <class IntType = int
          , class _BASE = std::geometric_distribution<IntType>
          >
class geometric_distribution : public _BASE {
public:
  geometric_distribution(float _parameter) : _BASE(_parameter), parameter(_parameter) {}

  const float parameter;

  /// get theoretical probability of a value
  template<typename value_type>
  double getProbability(value_type v) const {
    if (v<0) return 0.;
    return parameter * pow((1-parameter), v);
  }
};

}; // namespace test
}; // namespace AliceO2
#endif
