// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoDecay.h
/// \brief Implementation of the RecoDecay class.
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#ifndef O2_ANALYSIS_RECODECAY_H_
#define O2_ANALYSIS_RECODECAY_H_

#include <tuple>
#include <vector>
#include <array>
#include <cmath>

#include <TDatabasePDG.h>
#include <TPDGCode.h>

#include "CommonConstants/MathConstants.h"

using std::array;
using namespace o2::constants::math;

/// Base class for calculating properties of reconstructed decays.

class RecoDecay
{
 public:
  /// Default constructor
  RecoDecay() = default;

  /// Default destructor
  ~RecoDecay() = default;

  // Auxiliary functions

  /// Sums numbers.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return sum of numbers
  template <typename... T>
  static auto sum(const T&... args)
  {
    return (args + ...);
  }

  /// Squares a number.
  /// \note Promotes number to double before squaring to avoid precision loss in float multiplication.
  /// \param num  a number of arbitrary type
  /// \return number squared
  template <typename T>
  static double sq(T num)
  {
    return (double)num * (double)num;
  }

  /// Sums squares of numbers.
  /// \note Promotes numbers to double before squaring to avoid precision loss in float multiplication.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return sum of squares of numbers
  template <typename... T>
  static double sumOfSquares(const T&... args)
  {
    return (((double)args * (double)args) + ...);
  }

  /// Calculates square root of a sum of squares of numbers.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return square root of sum of squares of numbers
  template <typename... T>
  static double sqrtSumOfSquares(const T&... args)
  {
    return std::sqrt(sumOfSquares(args...));
  }

  /// Sums i-th elements of containers.
  /// \param index  element index
  /// \param args  pack of containers of elements accessible by index
  /// \return sum of i-th elements
  template <typename... T>
  static auto getElement(int index, const T&... args)
  {
    return (args[index] + ...);
  }

  /// Sums 3-vectors.
  /// \param args  pack of 3-vector arrays
  /// \return sum of vectors
  template <typename... T>
  static auto sumOfVec(const array<T, 3>&... args)
  {
    return array{getElement(0, args...), getElement(1, args...), getElement(2, args...)};
  }

  /// Calculates scalar product of vectors.
  /// \note Promotes numbers to double before squaring to avoid precision loss in float multiplication.
  /// \param N  dimension
  /// \param vec1,vec2  vectors
  /// \return scalar product
  template <std::size_t N, typename T, typename U>
  static double dotProd(const array<T, N>& vec1, const array<U, N>& vec2)
  {
    double res{0};
    for (auto iDim = 0; iDim < N; ++iDim) {
      res += (double)vec1[iDim] * (double)vec2[iDim];
    }
    return res;
  }

  /// Calculates magnitude squared of a vector.
  /// \param N  dimension
  /// \param vec  vector
  /// \return magnitude squared
  template <std::size_t N, typename T>
  static double mag2(const array<T, N>& vec)
  {
    return dotProd(vec, vec);
  }

  /// Calculates 3D distance between two points.
  /// \param point1,point2  {x, y, z} coordinates of points
  /// \return 3D distance between two points
  template <typename T, typename U>
  static double distance(const T& point1, const U& point2)
  {
    return sqrtSumOfSquares(point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]);
  }

  /// Calculates 2D {x, y} distance between two points.
  /// \param point1,point2  {x, y, z} or {x, y} coordinates of points
  /// \return 2D {x, y} distance between two points
  template <typename T, typename U>
  static double distanceXY(const T& point1, const U& point2)
  {
    return sqrtSumOfSquares(point1[0] - point2[0], point1[1] - point2[1]);
  }

  // Calculation of kinematic quantities

  /// Calculates pseudorapidity.
  /// \param mom  3-momentum array
  /// \return pseudorapidity
  template <typename T>
  static double Eta(const array<T, 3>& mom)
  {
    // eta = arctanh(pz/p)
    if (std::abs(mom[0]) < Almost0 && std::abs(mom[1]) < Almost0) { // very small px and py
      return (double)(mom[2] > 0 ? VeryBig : -VeryBig);
    }
    return (double)(std::atanh(mom[2] / P(mom)));
  }

  /// Calculates rapidity.
  /// \param mom  3-momentum array
  /// \param mass  mass
  /// \return rapidity
  template <typename T, typename U>
  static double Y(const array<T, 3>& mom, U mass)
  {
    // y = arctanh(pz/E)
    return std::atanh(mom[2] / E(mom, mass));
  }

  /// Calculates cosine of pointing angle.
  /// \param posPV  {x, y, z} position of the primary vertex
  /// \param posSV  {x, y, z} position of the secondary vertex
  /// \param mom  3-momentum array
  /// \return cosine of pointing angle
  template <typename T, typename U, typename V>
  static double CPA(const array<T, 3>& posPV, const array<U, 3>& posSV, const array<V, 3>& mom)
  {
    // CPA = (l . p)/(|l| |p|)
    auto lineDecay = array{posSV[0] - posPV[0], posSV[1] - posPV[1], posSV[2] - posPV[2]};
    auto cos = dotProd(lineDecay, mom) / std::sqrt(mag2(lineDecay) * mag2(mom));
    if (cos < -1.) {
      return -1.;
    }
    if (cos > 1.) {
      return 1.;
    }
    return cos;
  }

  /// Calculates cosine of pointing angle in the {x, y} plane.
  /// \param posPV  {x, y, z} or {x, y} position of the primary vertex
  /// \param posSV  {x, y, z} or {x, y} position of the secondary vertex
  /// \param mom  {x, y, z} or {x, y} momentum array
  /// \return cosine of pointing angle in {x, y}
  template <std::size_t N, std::size_t O, std::size_t P, typename T, typename U, typename V>
  static double CPAXY(const array<T, N>& posPV, const array<U, O>& posSV, const array<V, P>& mom)
  {
    // CPAXY = (r . pT)/(|r| |pT|)
    auto lineDecay = array{posSV[0] - posPV[0], posSV[1] - posPV[1]};
    auto momXY = array{mom[0], mom[1]};
    auto cos = dotProd(lineDecay, momXY) / std::sqrt(mag2(lineDecay) * mag2(momXY));
    if (cos < -1.) {
      return -1.;
    }
    if (cos > 1.) {
      return 1.;
    }
    return cos;
  }

  /// Calculates proper lifetime times c.
  /// \note Promotes numbers to double before squaring to avoid precision loss in float multiplication.
  /// \param mom  3-momentum array
  /// \param mass  mass
  /// \param length  decay length
  /// \return proper lifetime times c
  template <typename T, typename U, typename V>
  static double Ct(const array<T, 3>& mom, U length, V mass)
  {
    // c t = l m c^2/(p c)
    return (double)length * (double)mass / P(mom);
  }

  /// Calculates cosine of θ* (theta star).
  /// \note Implemented for 2 prongs only.
  /// \param arrMom  array of two 3-momentum arrays
  /// \param arrMass  array of two masses (in the same order as arrMom)
  /// \param mTot  assumed mass of mother particle
  /// \param iProng  index of the prong
  /// \return cosine of θ* of the i-th prong under the assumption of the invariant mass
  template <typename T, typename U, typename V>
  static double CosThetaStar(const array<array<T, 3>, 2>& arrMom, const array<U, 2>& arrMass, V mTot, int iProng)
  {
    auto pVecTot = PVec(arrMom[0], arrMom[1]);                                                                             // momentum of the mother particle
    auto pTot = P(pVecTot);                                                                                                // magnitude of the momentum of the mother particle
    auto eTot = E(pTot, mTot);                                                                                             // energy of the mother particle
    auto gamma = eTot / mTot;                                                                                              // γ, Lorentz gamma factor of the mother particle
    auto beta = pTot / eTot;                                                                                               // β, velocity of the mother particle
    auto pStar = std::sqrt(sq(sq(mTot) - sq(arrMass[0]) - sq(arrMass[1])) - sq(2 * arrMass[0] * arrMass[1])) / (2 * mTot); // p*, prong momentum in the rest frame of the mother particle
    // p* = √[(M^2 - m1^2 - m2^2)^2 - 4 m1^2 m2^2]/2M
    // Lorentz transformation of the longitudinal momentum of the prong into the detector frame:
    // p_L,i = γ (p*_L,i + β E*_i)
    // p*_L,i = p_L,i/γ - β E*_i
    // cos(θ*_i) = (p_L,i/γ - β E*_i)/p*
    return (dotProd(arrMom[iProng], pVecTot) / (pTot * gamma) - beta * E(pStar, arrMass[iProng])) / pStar;
  }

  /// Sums 3-momenta.
  /// \param args  pack of 3-momentum arrays
  /// \return total 3-momentum array
  template <typename... T>
  static auto PVec(const array<T, 3>&... args)
  {
    return sumOfVec(args...);
  }

  /// Calculates momentum squared from momentum components.
  /// \param px,py,pz  {x, y, z} momentum components
  /// \return momentum squared
  static double P2(double px, double py, double pz)
  {
    return sumOfSquares(px, py, pz);
  }

  /// Calculates total momentum squared of a sum of 3-momenta.
  /// \param args  pack of 3-momentum arrays
  /// \return total momentum squared
  template <typename... T>
  static double P2(const array<T, 3>&... args)
  {
    return sumOfSquares(getElement(0, args...), getElement(1, args...), getElement(2, args...));
  }

  /// Calculates (total) momentum magnitude.
  /// \param args  {x, y, z} momentum components or pack of 3-momentum arrays
  /// \return (total) momentum magnitude
  template <typename... T>
  static double P(const T&... args)
  {
    return std::sqrt(P2(args...));
  }

  /// Calculates transverse momentum squared from momentum components.
  /// \param px,py  {x, y} momentum components
  /// \return transverse momentum squared
  static double Pt2(double px, double py)
  {
    return sumOfSquares(px, py);
  }

  /// Calculates total transverse momentum squared of a sum of 3-(or 2-)momenta.
  /// \param args  pack of 3-(or 2-)momentum arrays
  /// \return total transverse momentum squared
  template <std::size_t N, typename... T>
  static double Pt2(const array<T, N>&... args)
  {
    return sumOfSquares(getElement(0, args...), getElement(1, args...));
  }

  /// Calculates (total) transverse momentum.
  /// \param args  {x, y} momentum components or pack of 3-momentum arrays
  /// \return (total) transverse momentum
  template <typename... T>
  static double Pt(const T&... args)
  {
    return std::sqrt(Pt2(args...));
  }

  /// Calculates energy squared from momentum and mass.
  /// \param args  momentum magnitude, mass
  /// \param args  {x, y, z} momentum components, mass
  /// \return energy squared
  template <typename... T>
  static double E2(T... args)
  {
    return sumOfSquares(args...);
  }

  /// Calculates energy squared from momentum vector and mass.
  /// \param mom  3-momentum array
  /// \param mass  mass
  /// \return energy squared
  template <typename T, typename U>
  static double E2(const array<T, 3>& mom, U mass)
  {
    return E2(mom[0], mom[1], mom[2], mass);
  }

  /// Calculates energy from momentum and mass.
  /// \param args  momentum magnitude, mass
  /// \param args  {x, y, z} momentum components, mass
  /// \param args  3-momentum array, mass
  /// \return energy
  template <typename... T>
  static double E(const T&... args)
  {
    return std::sqrt(E2(args...));
  }

  /// Calculates invariant mass squared from momentum magnitude and energy.
  /// \param mom  momentum magnitude
  /// \param energy  energy
  /// \return invariant mass squared
  static double M2(double mom, double energy)
  {
    return energy * energy - mom * mom;
  }

  /// Calculates invariant mass squared from momentum aray and energy.
  /// \param mom  3-momentum array
  /// \param energy  energy
  /// \return invariant mass squared
  template <typename T>
  static double M2(const array<T, 3>& mom, double energy)
  {
    return energy * energy - P2(mom);
  }

  /// Calculates invariant mass squared from momenta and masses of several particles (prongs).
  /// \param N  number of prongs
  /// \param arrMom  array of N 3-momentum arrays
  /// \param arrMass  array of N masses (in the same order as arrMom)
  /// \return invariant mass squared
  template <std::size_t N, typename T, typename U>
  static double M2(const array<array<T, 3>, N>& arrMom, const array<U, N>& arrMass)
  {
    array<double, 3> momTotal{0., 0., 0.}; // candidate momentum vector
    double energyTot{0.};                  // candidate energy
    for (auto iProng = 0; iProng < N; ++iProng) {
      for (auto iMom = 0; iMom < 3; ++iMom) {
        momTotal[iMom] += arrMom[iProng][iMom];
      } // loop over momentum components
      energyTot += E(arrMom[iProng], arrMass[iProng]);
    } // loop over prongs
    return energyTot * energyTot - P2(momTotal);
  }

  /// Calculates invariant mass.
  /// \param args  momentum magnitude, energy
  /// \param args  3-momentum array, energy
  /// \param args  array of momenta, array of masses
  /// \return invariant mass
  template <typename... T>
  static double M(const T&... args)
  {
    return std::sqrt(M2(args...));
  }

  /// Returns particle mass based on PDG code.
  /// \param pdg  PDG code
  /// \return particle mass
  static double getMassPDG(int pdg)
  {
    // Try to get the particle mass from the list first.
    for (const auto& particle : mListMass) {
      if (std::get<0>(particle) == pdg) {
        return std::get<1>(particle);
      }
    }
    // Get the mass of the new particle and add it in the list.
    auto newMass = TDatabasePDG::Instance()->GetParticle(pdg)->Mass();
    mListMass.push_back(std::make_tuple(pdg, newMass));
    return newMass;
  }

 private:
  static std::vector<std::tuple<int, double>> mListMass; ///< list of particle masses in form (PDG code, mass)
};

std::vector<std::tuple<int, double>> RecoDecay::mListMass;

#endif // O2_ANALYSIS_RECODECAY_H_
