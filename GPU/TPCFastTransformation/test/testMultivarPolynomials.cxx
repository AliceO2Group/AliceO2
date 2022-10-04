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

/// \file testMultivarPolynomials.cxx
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#define BOOST_TEST_MODULE Test TPC Fast Transformation
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MultivariatePolynomial.h"
#include "NDPiecewisePolynomials.h"
#include <vector>

namespace o2::gpu
{

// evaluate the polynomial of 4th degree in 5 dimensions for given coordinates and parameters
float evalPol4_5D(const float* x, const float* par)
{
  return par[0] + par[1] * x[0] + par[2] * x[1] + par[3] * x[2] + par[4] * x[3] + par[5] * x[4] + par[6] * x[0] * x[0] + par[7] * x[0] * x[1] + par[8] * x[0] * x[2] + par[9] * x[0] * x[3] + par[10] * x[0] * x[4] + par[11] * x[1] * x[1] + par[12] * x[1] * x[2] + par[13] * x[1] * x[3] + par[14] * x[1] * x[4] + par[15] * x[2] * x[2] + par[16] * x[2] * x[3] + par[17] * x[2] * x[4] + par[18] * x[3] * x[3] + par[19] * x[3] * x[4] + par[20] * x[4] * x[4] + par[21] * x[0] * x[0] * x[0] + par[22] * x[0] * x[0] * x[1] + par[23] * x[0] * x[0] * x[2] + par[24] * x[0] * x[0] * x[3] + par[25] * x[0] * x[0] * x[4] + par[26] * x[0] * x[1] * x[1] + par[27] * x[0] * x[1] * x[2] + par[28] * x[0] * x[1] * x[3] + par[29] * x[0] * x[1] * x[4] + par[30] * x[0] * x[2] * x[2] + par[31] * x[0] * x[2] * x[3] + par[32] * x[0] * x[2] * x[4] + par[33] * x[0] * x[3] * x[3] + par[34] * x[0] * x[3] * x[4] + par[35] * x[0] * x[4] * x[4] + par[36] * x[1] * x[1] * x[1] + par[37] * x[1] * x[1] * x[2] + par[38] * x[1] * x[1] * x[3] + par[39] * x[1] * x[1] * x[4] + par[40] * x[1] * x[2] * x[2] + par[41] * x[1] * x[2] * x[3] + par[42] * x[1] * x[2] * x[4] + par[43] * x[1] * x[3] * x[3] + par[44] * x[1] * x[3] * x[4] + par[45] * x[1] * x[4] * x[4] + par[46] * x[2] * x[2] * x[2] + par[47] * x[2] * x[2] * x[3] + par[48] * x[2] * x[2] * x[4] + par[49] * x[2] * x[3] * x[3] + par[50] * x[2] * x[3] * x[4] + par[51] * x[2] * x[4] * x[4] + par[52] * x[3] * x[3] * x[3] + par[53] * x[3] * x[3] * x[4] + par[54] * x[3] * x[4] * x[4] + par[55] * x[4] * x[4] * x[4] + par[56] * x[0] * x[0] * x[0] * x[0] + par[57] * x[0] * x[0] * x[0] * x[1] + par[58] * x[0] * x[0] * x[0] * x[2] + par[59] * x[0] * x[0] * x[0] * x[3] + par[60] * x[0] * x[0] * x[0] * x[4] + par[61] * x[0] * x[0] * x[1] * x[1] + par[62] * x[0] * x[0] * x[1] * x[2] + par[63] * x[0] * x[0] * x[1] * x[3] + par[64] * x[0] * x[0] * x[1] * x[4] + par[65] * x[0] * x[0] * x[2] * x[2] + par[66] * x[0] * x[0] * x[2] * x[3] + par[67] * x[0] * x[0] * x[2] * x[4] + par[68] * x[0] * x[0] * x[3] * x[3] + par[69] * x[0] * x[0] * x[3] * x[4] + par[70] * x[0] * x[0] * x[4] * x[4] + par[71] * x[0] * x[1] * x[1] * x[1] + par[72] * x[0] * x[1] * x[1] * x[2] + par[73] * x[0] * x[1] * x[1] * x[3] + par[74] * x[0] * x[1] * x[1] * x[4] + par[75] * x[0] * x[1] * x[2] * x[2] + par[76] * x[0] * x[1] * x[2] * x[3] + par[77] * x[0] * x[1] * x[2] * x[4] + par[78] * x[0] * x[1] * x[3] * x[3] + par[79] * x[0] * x[1] * x[3] * x[4] + par[80] * x[0] * x[1] * x[4] * x[4] + par[81] * x[0] * x[2] * x[2] * x[2] + par[82] * x[0] * x[2] * x[2] * x[3] + par[83] * x[0] * x[2] * x[2] * x[4] + par[84] * x[0] * x[2] * x[3] * x[3] + par[85] * x[0] * x[2] * x[3] * x[4] + par[86] * x[0] * x[2] * x[4] * x[4] + par[87] * x[0] * x[3] * x[3] * x[3] + par[88] * x[0] * x[3] * x[3] * x[4] + par[89] * x[0] * x[3] * x[4] * x[4] + par[90] * x[0] * x[4] * x[4] * x[4] + par[91] * x[1] * x[1] * x[1] * x[1] + par[92] * x[1] * x[1] * x[1] * x[2] + par[93] * x[1] * x[1] * x[1] * x[3] + par[94] * x[1] * x[1] * x[1] * x[4] + par[95] * x[1] * x[1] * x[2] * x[2] + par[96] * x[1] * x[1] * x[2] * x[3] + par[97] * x[1] * x[1] * x[2] * x[4] + par[98] * x[1] * x[1] * x[3] * x[3] + par[99] * x[1] * x[1] * x[3] * x[4] + par[100] * x[1] * x[1] * x[4] * x[4] + par[101] * x[1] * x[2] * x[2] * x[2] + par[102] * x[1] * x[2] * x[2] * x[3] + par[103] * x[1] * x[2] * x[2] * x[4] + par[104] * x[1] * x[2] * x[3] * x[3] + par[105] * x[1] * x[2] * x[3] * x[4] + par[106] * x[1] * x[2] * x[4] * x[4] + par[107] * x[1] * x[3] * x[3] * x[3] + par[108] * x[1] * x[3] * x[3] * x[4] + par[109] * x[1] * x[3] * x[4] * x[4] + par[110] * x[1] * x[4] * x[4] * x[4] + par[111] * x[2] * x[2] * x[2] * x[2] + par[112] * x[2] * x[2] * x[2] * x[3] + par[113] * x[2] * x[2] * x[2] * x[4] + par[114] * x[2] * x[2] * x[3] * x[3] + par[115] * x[2] * x[2] * x[3] * x[4] + par[116] * x[2] * x[2] * x[4] * x[4] + par[117] * x[2] * x[3] * x[3] * x[3] + par[118] * x[2] * x[3] * x[3] * x[4] + par[119] * x[2] * x[3] * x[4] * x[4] + par[120] * x[2] * x[4] * x[4] * x[4] + par[121] * x[3] * x[3] * x[3] * x[3] + par[122] * x[3] * x[3] * x[3] * x[4] + par[123] * x[3] * x[3] * x[4] * x[4] + par[124] * x[3] * x[4] * x[4] * x[4] + par[125] * x[4] * x[4] * x[4] * x[4];
}

float evalPol5_5D_interactionOnly(const float* x, const float* par)
{
  return par[0] + x[0] * par[1] + x[1] * par[2] + x[2] * par[3] + x[3] * par[4] + x[4] * par[5] + x[0] * x[1] * par[6] + x[0] * x[2] * par[7] + x[0] * x[3] * par[8] + x[0] * x[4] * par[9] + x[1] * x[2] * par[10] + x[1] * x[3] * par[11] + x[1] * x[4] * par[12] + x[2] * x[3] * par[13] + x[2] * x[4] * par[14] + x[3] * x[4] * par[15] + x[0] * x[1] * x[2] * par[16] + x[0] * x[1] * x[3] * par[17] + x[0] * x[1] * x[4] * par[18] + x[0] * x[2] * x[3] * par[19] + x[0] * x[2] * x[4] * par[20] + x[0] * x[3] * x[4] * par[21] + x[1] * x[2] * x[3] * par[22] + x[1] * x[2] * x[4] * par[23] + x[1] * x[3] * x[4] * par[24] + x[2] * x[3] * x[4] * par[25] + x[0] * x[1] * x[2] * x[3] * par[26] + x[0] * x[1] * x[2] * x[4] * par[27] + x[0] * x[1] * x[3] * x[4] * par[28] + x[0] * x[2] * x[3] * x[4] * par[29] + x[1] * x[2] * x[3] * x[4] * par[30] + x[0] * x[1] * x[2] * x[3] * x[4] * par[31];
}

float genRand()
{
  const float minVal = -5;
  const float maxVal = 5;
  const float val = minVal + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxVal - minVal)));
  return val;
}

BOOST_AUTO_TEST_CASE(Polynomials5D)
{
  std::srand(std::time(nullptr));
  const int nPar5D4Deg = 126;         // number of parameters
  const int nDim = 5;                 // dimensions
  const int nDegree = 4;              // degree
  const float abstolerance = 0.0001f; // abosulte difference between refernce to polynomial class

  MultivariatePolynomial<nDim, nDegree> polCT;       // compile time polynomial
  MultivariatePolynomial<0, 0> polRT(nDim, nDegree); // run time polynomial

  // compare number of parameters
  BOOST_CHECK(nPar5D4Deg == polCT.getNParams());
  BOOST_CHECK(nPar5D4Deg == polRT.getNParams());

  float par[nPar5D4Deg]{20};
  for (int iter = 0; iter < 10; ++iter) {

    // draw random parameters
    for (int i = 1; i < nPar5D4Deg; ++i) {
      par[i] = genRand();
    }

    polCT.setParams(par);
    polRT.setParams(par);

    // compare evaluated polynomials
    for (float a = 0; a < 1; a += 0.2f) {
      for (float b = 0; b < 1; b += 0.2f) {
        for (float c = 0; c < 1; c += 0.2f) {
          for (float d = 0; d < 1; d += 0.2f) {
            for (float e = 0; e < 1; e += 0.2f) {
              const float arr[nDim]{a, b, c, d, e};
              const float valCT = polCT.eval(arr);
              const float valRT = polRT.eval(arr);
              const float valRef = evalPol4_5D(arr, par);
              BOOST_CHECK_SMALL(valCT - valRef, abstolerance);
              BOOST_CHECK_SMALL(valRT - valRef, abstolerance);
            }
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(Polynomials5D_InteractionOnly)
{
  std::srand(std::time(nullptr));
  const int nPar5D5DegInteraction = 32; // number of parameters
  const int nDim = 5;                   // dimensions
  const int nDegree = 5;                // degree
  const float abstolerance = 0.0001f;   // abosulte difference between refernce to polynomial class
  const bool interactionOnly = true;

  MultivariatePolynomial<nDim, nDegree, interactionOnly> polCT;       // compile time polynomial
  MultivariatePolynomial<0, 0> polRT(nDim, nDegree, interactionOnly); // run time polynomial

  // compare number of parameters
  BOOST_CHECK(nPar5D5DegInteraction == polCT.getNParams());
  BOOST_CHECK(nPar5D5DegInteraction == polRT.getNParams());

  float par[nPar5D5DegInteraction]{20};
  for (int iter = 0; iter < 10; ++iter) {

    // draw random parameters
    for (int i = 1; i < nPar5D5DegInteraction; ++i) {
      par[i] = genRand();
    }

    polCT.setParams(par);
    polRT.setParams(par);

    // compare evaluated polynomials
    for (float a = 0; a < 1; a += 0.2f) {
      for (float b = 0; b < 1; b += 0.2f) {
        for (float c = 0; c < 1; c += 0.2f) {
          for (float d = 0; d < 1; d += 0.2f) {
            for (float e = 0; e < 1; e += 0.2f) {
              const float arr[nDim]{a, b, c, d, e};
              const float valCT = polCT.eval(arr);
              const float valRT = polRT.eval(arr);
              const float valRef = evalPol5_5D_interactionOnly(arr, par);
              BOOST_CHECK_SMALL(valCT - valRef, abstolerance);
              BOOST_CHECK_SMALL(valRT - valRef, abstolerance);
            }
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(Piecewise_polynomials)
{
  std::srand(std::time(nullptr));
  const int nPar5D5DegInteraction = 32; // number of parameters
  const int nDim = 5;                   // dimensions
  const int nDegree = 5;                // degree
  const bool interactionOnly = true;    // consider only interaction terms

  // reference polynomial which will be approximated by the NDPiecewisePolynomials
  MultivariatePolynomial<nDim, nDegree, interactionOnly> polCT;

  // define picewise polynomials with a lower degree than the original reference function
  std::vector<float> xMin(nDim, 0);
  std::vector<float> xMax(nDim, 1);
  const std::vector<unsigned int> nVert(nDim, 4);
  NDPiecewisePolynomials<nDim, nDegree - 2, interactionOnly> piecewisePol(xMin.data(), xMax.data(), nVert.data());

  // define function which will be used to approximate the picewise polynomials
  std::function<float(const double x[/* Xdim */])> refFunc = [&polCT, nDim](const double x[/* Xdim */]) {
    float val[nDim]{};
    std::copy(x, x + nDim, val);
    return polCT.eval(val);
  };

  float par[nPar5D5DegInteraction]{20};
  for (int iter = 0; iter < 5; ++iter) {

    // draw random parameters
    for (int i = 1; i < nPar5D5DegInteraction; ++i) {
      par[i] = genRand();
    }

    polCT.setParams(par);

    // define number of axiliary points which will be used for the fitting
    const unsigned int nAux = 4;
    const std::vector<unsigned int> nAxiliaryPoints(nDim, nAux);

    // perform the fitting of the picewise polynomials
    piecewisePol.performFits(refFunc, nAxiliaryPoints.data());

    // compare evaluated polynomials
    for (float a = 0; a < 1; a += 0.2f) {
      for (float b = 0; b < 1; b += 0.2f) {
        for (float c = 0; c < 1; c += 0.2f) {
          for (float d = 0; d < 1; d += 0.2f) {
            for (float e = 0; e < 1; e += 0.2f) {
              const float arr[nDim]{a, b, c, d, e};
              const float valCT = polCT.eval(arr);
              const float valpiecewisePol = piecewisePol.evalUnsafe(arr);
              if (std::abs(valCT) < 0.2) {
                BOOST_CHECK_SMALL(valCT - valpiecewisePol, 0.02f);
              } else {
                BOOST_CHECK_CLOSE(valpiecewisePol, valCT, 0.5);
              }
            }
          }
        }
      }
    }

    // check for safe evaluation outside of the grid at lower boundary
    for (float a = xMin[0] - 10; a <= xMin[0]; a += 2.f) {
      for (float b = xMin[1] - 10; b <= xMin[1]; b += 2.f) {
        for (float c = xMin[2] - 10; c <= xMin[2]; c += 2.f) {
          for (float d = xMin[3] - 10; d <= xMin[3]; d += 2.f) {
            for (float e = xMin[4] - 10; e <= xMin[4]; e += 2.f) {
              float arr[nDim]{a, b, c, d, e};
              const float valUnsafeLow = piecewisePol.evalUnsafe(xMin.data());
              const float valSafeLow = piecewisePol.eval(arr);
              if (std::abs(valUnsafeLow) < 0.1) {
                BOOST_CHECK_SMALL(valSafeLow - valUnsafeLow, 0.0001f);
              } else {
                BOOST_CHECK_CLOSE(valSafeLow, valUnsafeLow, 0.001);
              }
            }
          }
        }
      }
    }

    // check for safe evaluation outside of the grid at upper boundary
    const std::vector<int> indexHigh(nDim, nAux - 2);
    for (float a = xMax[0]; a <= xMax[0] + 10; a += 2.f) {
      for (float b = xMax[1]; b <= xMax[1] + 10; b += 2.f) {
        for (float c = xMax[2]; c <= xMax[2] + 10; c += 2.f) {
          for (float d = xMax[3]; d <= xMax[3] + 10; d += 2.f) {
            for (float e = xMax[4]; e <= xMax[4] + 10; e += 2.f) {
              float arr[nDim]{a, b, c, d, e};
              const float valUnsafeHigh = piecewisePol.evalPol(xMax.data(), indexHigh.data());
              const float valSafeHigh = piecewisePol.eval(arr);
              if (std::abs(valUnsafeHigh) < 0.1) {
                BOOST_CHECK_SMALL(valSafeHigh - valUnsafeHigh, 0.0001f);
              } else {
                BOOST_CHECK_CLOSE(valSafeHigh, valUnsafeHigh, 0.001);
              }
            }
          }
        }
      }
    }
  }
}

} // namespace o2::gpu
