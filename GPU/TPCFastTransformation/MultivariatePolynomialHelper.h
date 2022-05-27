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

/// \file MultivariatePolynomialHelper.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_MULTIVARIATEPOLYNOMIALHELPER
#define ALICEO2_TPC_MULTIVARIATEPOLYNOMIALHELPER

#include "GPUCommonDef.h"

#if !defined(GPUCA_GPUCODE)
#include <vector>
#include <string>
#include <cassert>
#if !defined(GPUCA_NO_FMT)
#include <fmt/format.h>
#endif
#endif

namespace GPUCA_NAMESPACE::gpu
{

#if !defined(GPUCA_GPUCODE)

/// simple struct to enable writing the MultivariatePolynomial to file
struct MultivariatePolynomialContainer {

  /// constructor
  /// \param dim number of dimensions of the polynomial
  /// \param degree degree of the polynomials
  /// \param nParameters number of parameters
  /// \param params parmaeters
  MultivariatePolynomialContainer(const unsigned int dim, const unsigned int degree, const unsigned int nParameters, const float params[/* nParameters*/]) : mDim{dim}, mDegree{degree}, mParams{params, params + nParameters} {};

  /// for ROOT I/O
  MultivariatePolynomialContainer() = default;

  const unsigned int mDim{};          ///< number of dimensions of the polynomial
  const unsigned int mDegree{};       ///< degree of the polynomials
  const std::vector<float> mParams{}; ///< parameters of the polynomial
};
#endif

/// Helper class for calculating the number of parameters for a multidimensional polynomial
class MultivariatePolynomialParametersHelper
{
 public:
  /// \returns number of parameters for given dimension and degree of polynomials
  /// calculates the number of parameters for a multivariate polynomial for given degree: nParameters = (n+d-1 d) -> binomial coefficient
  /// see: https://mathoverflow.net/questions/225953/number-of-polynomial-terms-for-certain-degree-and-certain-number-of-variables
  GPUd() static constexpr unsigned int getNParameters(const unsigned int degree, const unsigned int dim) { return (degree == 0) ? binomialCoeff(dim - 1, 0) : binomialCoeff(dim - 1 + degree, degree) + getNParameters(degree - 1, dim); }

 private:
  /// calculate factorial of n
  /// \return returns n!
  GPUd() static constexpr unsigned int factorial(const unsigned int n) { return (n == 0) || (n == 1) ? 1 : n * factorial(n - 1); }

  /// calculates binomial coefficient
  /// \return returns (n k)
  GPUd() static constexpr unsigned int binomialCoeff(const unsigned int n, const unsigned int k) { return factorial(n) / (factorial(k) * factorial(n - k)); }
};

/// Helper struct for evaluating a multidimensional polynomial using compile time evaluated formula
/// Compile time method to extract the formula is obtained from run time method (check combination_with_repetiton() and evalPol())
/// by performing all loops during compile time and replacing the array to keep track of the dimensions for given term (pos[FMaxdegree + 1])
/// to a simple unsigned int called Pos where each digit represents the dimension for a given term e.g. pos = 2234 -> x[2]*x[2]*x[3]*x[4]
///
template <unsigned int Dim, unsigned int Degree>
class MultivariatePolynomialHelper : public MultivariatePolynomialParametersHelper
{
  static constexpr unsigned short FMaxdim = 10;   ///< maximum dimensionality of the polynomials (number of different digits: 0,1,2,3....9 )
  static constexpr unsigned short FMaxdegree = 9; ///< maximum degree of the polynomials (maximum number of digits in unsigned integer - 1)

#if !defined(GPUCA_GPUCODE)
  static_assert(Dim <= MultivariatePolynomialHelper<Dim, Degree>::FMaxdim && Degree <= MultivariatePolynomialHelper<Dim, Degree>::FMaxdegree, "Max. number of dimensions or degrees exceeded!");
#endif

 public:
  /// evaluates the polynomial for given parameters and coordinates
  /// \param par parameters of the polynomials
  /// \param x input coordinates
  GPUd() static constexpr float evalPol(GPUgeneric() const float par[/*number of parameters*/], const float x[/*number of dimensions*/]) { return par[0] + loopDegrees<1>(par, x); }

  /// \return returns number of dimensions of the polynomials
  GPUd() static constexpr unsigned int getDim() { return Dim; }

  /// \return returns the degree of the polynomials
  GPUd() static constexpr unsigned int getDegree() { return Degree; }

 private:
  /// computes power of 10
  GPUd() static constexpr unsigned int pow10(const unsigned int n) { return n == 0 ? 1 : 10 * pow10(n - 1); }

  /// helper for modulo to extract the digit in an integer a at position b (can be obtained with pow10(digitposition)): e.g. a=1234 b=pow10(2)=100 -> returns 2
  GPUd() static constexpr unsigned int mod10(const unsigned int a, const unsigned int b) { return (a / b) % 10; }

  /// resetting digits of pos for given position to refDigit
  GPUd() static constexpr unsigned int resetIndices(const unsigned int degreePol, const unsigned int pos, const unsigned int leftDigit, const unsigned int iter, const unsigned int refDigit);

  GPUd() static constexpr unsigned int getNewPos(const unsigned int degreePol, const unsigned int pos, const unsigned int digitPos);

  /// calculates term e.g. x^3*y
  /// \tparam DegreePol max degree of the polynomials
  /// \pos decoded information about the current term e.g. 1233 -> x[1]*x[2]*x[3]*x[3] (otherwise an array could be used)
  template <unsigned int DegreePol>
  GPUd() static constexpr float prodTerm(const float x[], const unsigned int pos);

  /// calculate sum of the terms for given degree -> summation of the par[]*x^4 + par[]*x^3*y + par[]*x^3*z + par[]*x^2*y^2..... terms
  /// \tparam DegreePol max degree of the polynomials
  /// \Pos decoded information about the current term e.g. 1233 -> x[1]*x[2]*x[3]*x[3] (otherwise an array could be used)
  /// \tparam Index index for accessing the parameters
  template <unsigned int DegreePol, unsigned int Pos, unsigned int Index>
  GPUd() static constexpr float sumTerms(GPUgeneric() const float par[], const float x[]);

  /// loop over the degrees of the polynomials (for formula see https://math.stackexchange.com/questions/1234240/equation-that-defines-multi-dimensional-polynomial)
  /// \tparam degree iteration of the loop which starts from 1 to the max degree of the polynomial e.g. Iter=4 -> summation of the par[]*x^4 + par[]*x^3*y + par[]*x^3*z + par[]*x^2*y^2..... terms
  /// \param par parameters of the pokynomial
  /// \param x input coordinates
  template <unsigned int DegreePol>
  GPUd() static constexpr float loopDegrees(GPUgeneric() const float par[], const float x[]);
};

/// Helper struct for evaluating a multidimensional polynomial using run time evaluated formula
template <>
class MultivariatePolynomialHelper<0, 0> : public MultivariatePolynomialParametersHelper
{
 public:
#if !defined(GPUCA_GPUCODE)
  /// constructor
  /// \param nDim dimensionality of the polynomials
  /// \param degree degree of the polynomials
  MultivariatePolynomialHelper(const unsigned int nDim, const unsigned int degree) : mDim{nDim}, mDegree{degree} { assert(mDegree <= FMaxdegree); };
#endif

  /// default constructor
  MultivariatePolynomialHelper() CON_DEFAULT;

  /// Destructor
  ~MultivariatePolynomialHelper() CON_DEFAULT;

#if !defined(GPUCA_GPUCODE)
  /// printing the formula of the polynomial
  void print() const;

  /// \return returns the terms which are used to evaluate the polynomial
  std::vector<std::string> getTerms() const;
#endif

  /// evaluating the polynomial
  /// \param par coefficients of the polynomial
  /// \param x input coordinates
  GPUd() float evalPol(GPUgeneric() const float par[/*number of parameters*/], const float x[/*number of dimensions*/]) const { return evalPol(par, x, mDegree, mDim); }

  /// \return returns number of dimensions of the polynomials
  GPUd() unsigned int getDim() const { return mDim; }

  /// \return returns the degree of the polynomials
  GPUd() unsigned int getDegree() const { return mDegree; }

 protected:
  unsigned int mDim{};    ///< dimensionality of the polynomial
  unsigned int mDegree{}; ///< maximum degree of the polynomial

 private:
  static constexpr unsigned short FMaxdegree = 9; ///< maximum degree of the polynomials (can be increased if desired: size of array in combination_with_repetiton: pos[FMaxdegree + 1])

  /// evalutes the polynomial
  GPUd() static constexpr float evalPol(GPUgeneric() const float par[], const float x[], const unsigned int degree, const unsigned int dim);

  /// helper function to get all combinations
  template <class Type>
  GPUd() static Type combination_with_repetiton(const unsigned int degree, const unsigned int dim, GPUgeneric() const float par[], int& indexPar, const float x[]);
  GPUd() static constexpr float combination_with_repetiton_constexpr_float(const unsigned int degree, const unsigned int dim, GPUgeneric() const float par[], int& indexPar, const float x[]);
};

//=================================================================================
//============================ inline implementations =============================
//=================================================================================

template <unsigned int Dim, unsigned int Degree>
GPUd() constexpr unsigned int MultivariatePolynomialHelper<Dim, Degree>::resetIndices(const unsigned int degreePol, const unsigned int pos, const unsigned int leftDigit, const unsigned int iter, const unsigned int refDigit)
{
  if (iter <= degreePol) {
    const int powTmp = pow10(leftDigit);
    const int rightDigit = mod10(pos, powTmp);
    const int posTmp = pos - (rightDigit - refDigit) * powTmp;
    return resetIndices(degreePol, posTmp, leftDigit - 1, iter + 1, refDigit);
  }
  return pos;
}

template <unsigned int Dim, unsigned int Degree>
GPUd() constexpr unsigned int MultivariatePolynomialHelper<Dim, Degree>::getNewPos(const unsigned int degreePol, const unsigned int pos, const unsigned int digitPos)
{
  if (degreePol > digitPos) {
    // check if digit of current position is at is max position
    if (mod10(pos, pow10(digitPos)) == Dim) {
      // increase digit of left position
      const unsigned int leftDigit = digitPos + 1;
      const unsigned int posTmp = pos + pow10(leftDigit);
      const unsigned int refDigit = mod10(posTmp, pow10(digitPos + 1));

      // resetting digits to the right if digit exceeds number of dimensions
      const unsigned int posReset = resetIndices(degreePol, posTmp, leftDigit - 1, degreePol - digitPos, refDigit);

      // check next digit
      return getNewPos(degreePol, posReset, digitPos + 1);
    }
    return getNewPos(degreePol, pos, digitPos + 1);
  }
  return pos;
}

template <unsigned int Dim, unsigned int Degree>
template <unsigned int DegreePol>
GPUd() constexpr float MultivariatePolynomialHelper<Dim, Degree>::prodTerm(const float x[], const unsigned int pos)
{
  if constexpr (DegreePol > 0) {
    // extract index of the dimension which is decoded in the digit
    const unsigned int index = mod10(pos, pow10(DegreePol - 1));
    return x[index] * prodTerm<DegreePol - 1>(x, pos);
  }
  return 1;
}

template <unsigned int Dim, unsigned int Degree>
template <unsigned int DegreePol, unsigned int Pos, unsigned int Index>
GPUd() constexpr float MultivariatePolynomialHelper<Dim, Degree>::sumTerms(GPUgeneric() const float par[], const float x[])
{
  // checking if the current position is reasonable e.g. if the max dimension is x[4]: for Pos=15 -> x[1]*x[5] the position is set to 22 -> x[2]*x[2]
  constexpr unsigned int posNew = getNewPos(DegreePol, Pos, 0);
  if constexpr (mod10(posNew, pow10(DegreePol)) != 1) {
    // sum up the term for corrent term and set posotion for next combination
    return par[Index] * prodTerm<DegreePol>(x, posNew) + sumTerms<DegreePol, posNew + 1, Index + 1>(par, x);
  }
  return 0;
}

template <unsigned int Dim, unsigned int Degree>
template <unsigned int DegreePol>
GPUd() constexpr float MultivariatePolynomialHelper<Dim, Degree>::loopDegrees(GPUgeneric() const float par[], const float x[])
{
  if constexpr (DegreePol <= Degree) {
    constexpr unsigned int index{getNParameters(DegreePol - 1, Dim)}; // offset of the index for accessing the parameters
    return sumTerms<DegreePol, 0, index>(par, x) + loopDegrees<DegreePol + 1>(par, x);
  }
  return 0;
}

template <class Type>
GPUd() Type MultivariatePolynomialHelper<0, 0>::combination_with_repetiton(const unsigned int degree, const unsigned int dim, GPUgeneric() const float par[], int& indexPar, const float x[])
{
  {
    const unsigned int size = degree + 1;
    unsigned int pos[FMaxdegree + 1]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // return value is either the sum of all polynomials or a vector of strings containing the formula for each polynomial
    Type val(0);
    for (;;) {
      // starting on the rightmost digit
      for (unsigned int i = degree; i > 0; --i) {
        // check if digit of current position is at is max position
        if (pos[i] == dim) {
          // increase digit of left position
          ++pos[i - 1];
          // resetting the indices of the digits to the right
          for (unsigned int j = i; j <= degree; ++j) {
            pos[j] = pos[i - 1];
          }
        }
      }

      // check if all combinations are processed
      if (pos[0] == 1) {
        break;
      } else {
        if constexpr (std::is_same_v<Type, float>) {
          float term = par[indexPar++];
          for (size_t i = 1; i < size; ++i) {
            term *= x[pos[i]];
          }
          val += term;
        } else {
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_FMT) && !defined(GPUCA_ALIROOT_LIB)
          std::string term{};
          for (size_t i = 1; i < size; ++i) {
            term += fmt::format("x[{}] * ", pos[i]);
          }
          term += fmt::format("par[{}]", indexPar++);
          val.emplace_back(term);
#endif
        }
      }
      // increase the rightmost digit
      ++pos[degree];
    }
    return val;
  }
}

GPUd() constexpr float MultivariatePolynomialHelper<0, 0>::combination_with_repetiton_constexpr_float(const unsigned int degree, const unsigned int dim, GPUgeneric() const float par[], int& indexPar, const float x[])
{
  {
    const unsigned int size = degree + 1;
    unsigned int pos[FMaxdegree + 1]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // return value is either the sum of all polynomials or a vector of strings containing the formula for each polynomial
    float val(0);
    for (;;) {
      // starting on the rightmost digit
      for (unsigned int i = degree; i > 0; --i) {
        // check if digit of current position is at is max position
        if (pos[i] == dim) {
          // increase digit of left position
          ++pos[i - 1];
          // resetting the indices of the digits to the right
          for (unsigned int j = i; j <= degree; ++j) {
            pos[j] = pos[i - 1];
          }
        }
      }

      // check if all combinations are processed
      if (pos[0] == 1) {
        break;
      } else {
        if constexpr (std::is_same_v<float, float>) {
          float term = par[indexPar++];
          for (size_t i = 1; i < size; ++i) {
            term *= x[pos[i]];
          }
          val += term;
        } else {
          // Removed since it cannot work as constexpr
          /*std::string term{};
          for (size_t i = 1; i < size; ++i) {
            term += fmt::format("x[{}] * ", pos[i]);
          }
          term += fmt::format("par[{}]", indexPar++);
          val.emplace_back(term);*/
        }
      }
      // increase the rightmost digit
      ++pos[degree];
    }
    return val;
  }
}

GPUd() constexpr float MultivariatePolynomialHelper<0, 0>::evalPol(GPUgeneric() const float par[], const float x[], const unsigned int degree, const unsigned int dim)
{
  float val = par[0];
  int indexPar = 1;
  for (unsigned int deg = 1; deg <= degree; ++deg) {
    val += combination_with_repetiton_constexpr_float(deg, dim, par, indexPar, x);
  }
  return val;
}

} // namespace GPUCA_NAMESPACE::gpu

#endif
