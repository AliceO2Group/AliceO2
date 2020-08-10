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

#ifndef RECODECAY_H
#define RECODECAY_H

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

  /// Sums squares of numbers.
  /// \note Promotes numbers to double before squaring to avoid precision loss in float multiplication.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return sum of squares of numbers
  template <typename... T>
  static auto sumOfSquares(const T&... args)
  {
    return (((double)args * (double)args) + ...);
  }

  /// Calculates square root of a sum of squares of numbers.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return square root of sum of squares of numbers
  template <typename... T>
  static auto sqrtSumOfSquares(const T&... args)
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
  static auto dotProd(const array<T, N>& vec1, const array<U, N>& vec2)
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
  static auto mag2(const array<T, N>& vec)
  {
    return dotProd(vec, vec);
  }

  /// Calculates 3D distance between two points.
  /// \param point1,point2  {x, y, z} coordinates of points
  /// \return 3D distance between two points
  template <typename T, typename U>
  static auto distance(const array<T, 3>& point1, const array<U, 3>& point2)
  {
    return sqrtSumOfSquares(point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]);
  }

  /// Calculates 2D {x, y} distance between two points.
  /// \param point1,point2  {x, y, z} or {x, y} coordinates of points
  /// \return 2D {x, y} distance between two points
  template <std::size_t N, std::size_t O, typename T, typename U>
  static auto distanceXY(const array<T, N>& point1, const array<U, O>& point2)
  {
    return sqrtSumOfSquares(point1[0] - point2[0], point1[1] - point2[1]);
  }

  // Calculation of kinematic quantities

  /// Calculates pseudorapidity.
  /// \param mom  3-momentum array
  /// \return pseudorapidity
  template <typename T>
  static auto Eta(const array<T, 3>& mom)
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
  static auto Rapidity(const array<T, 3>& mom, U mass)
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
  static auto CPA(const array<T, 3>& posPV, const array<U, 3>& posSV, const array<V, 3>& mom)
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
  static auto CPAXY(const array<T, N>& posPV, const array<U, O>& posSV, const array<V, P>& mom)
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
  static auto Ct(array<T, 3> mom, U mass, V length)
  {
    // c t = l m c^2/(p c)
    return (double)length * (double)mass / P(mom);
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
  static auto P2(double px, double py, double pz)
  {
    return sumOfSquares(px, py, pz);
  }

  /// Calculates total momentum squared of a sum of 3-momenta.
  /// \param args  pack of 3-momentum arrays
  /// \return total momentum squared
  template <typename... T>
  static auto P2(const array<T, 3>&... args)
  {
    return sumOfSquares(getElement(0, args...), getElement(1, args...), getElement(2, args...));
  }

  /// Calculates (total) momentum magnitude.
  /// \param args  {x, y, z} momentum components or pack of 3-momentum arrays
  /// \return (total) momentum magnitude
  template <typename... T>
  static auto P(const T&... args)
  {
    return std::sqrt(P2(args...));
  }

  /// Calculates transverse momentum squared from momentum components.
  /// \param px,py  {x, y} momentum components
  /// \return transverse momentum squared
  static auto Pt2(double px, double py)
  {
    return sumOfSquares(px, py);
  }

  /// Calculates total transverse momentum squared of a sum of 3-(or 2-)momenta.
  /// \param args  pack of 3-(or 2-)momentum arrays
  /// \return total transverse momentum squared
  template <std::size_t N, typename... T>
  static auto Pt2(const array<T, N>&... args)
  {
    return sumOfSquares(getElement(0, args...), getElement(1, args...));
  }

  /// Calculates (total) transverse momentum.
  /// \param args  {x, y} momentum components or pack of 3-momentum arrays
  /// \return (total) transverse momentum
  template <typename... T>
  static auto Pt(const T&... args)
  {
    return std::sqrt(Pt2(args...));
  }

  /// Calculates energy squared from momentum and mass.
  /// \param args  momentum magnitude, mass
  /// \param args  {x, y, z} momentum components, mass
  /// \return energy squared
  template <typename... T>
  static auto E2(T... args)
  {
    return sumOfSquares(args...);
  }

  /// Calculates energy squared from momentum vector and mass.
  /// \param mom  3-momentum array
  /// \param mass  mass
  /// \return energy squared
  template <typename T, typename U>
  static auto E2(const array<T, 3>& mom, U mass)
  {
    return E2(mom[0], mom[1], mom[2], mass);
  }

  /// Calculates energy from momentum and mass.
  /// \param args  momentum magnitude, mass
  /// \param args  {x, y, z} momentum components, mass
  /// \param args  3-momentum array, mass
  /// \return energy
  template <typename... T>
  static auto E(T... args)
  {
    return std::sqrt(E2(args...));
  }

  /// Calculates invariant mass squared from momentum magnitude and energy.
  /// \param mom  momentum magnitude
  /// \param energy  energy
  /// \return invariant mass squared
  static auto M2(double mom, double energy)
  {
    return energy * energy - mom * mom;
  }

  /// Calculates invariant mass squared from momentum aray and energy.
  /// \param mom  3-momentum array
  /// \param energy  energy
  /// \return invariant mass squared
  template <typename T>
  static auto M2(const array<T, 3>& mom, double energy)
  {
    return energy * energy - P2(mom);
  }

  /// Calculates invariant mass squared from momenta and masses of several particles (prongs).
  /// \param N  number of prongs
  /// \param arrMom  array of N 3-momentum arrays
  /// \param arrMass  array of N masses (in the same order as arrMom)
  /// \return invariant mass squared
  template <std::size_t N, typename T, typename U>
  static auto M2(const array<array<T, 3>, N>& arrMom, const array<U, N>& arrMass)
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
  static auto M(T... args)
  {
    return std::sqrt(M2(args...));
  }

  /// Returns particle mass based on PDG code.
  /// \param pdg  PDG code
  /// \return particle mass
  static auto getMassPDG(int pdg)
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

#endif // RECODECAY_H
