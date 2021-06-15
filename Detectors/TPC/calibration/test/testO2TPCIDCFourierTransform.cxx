// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  testO2TPCIDCFourierTransform.cxx
/// \brief this task tests the calculation of fourier coefficients by comparing input values with FT->IFT values
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#define BOOST_TEST_MODULE Test TPC O2TPCIDCFourierTransform class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCCalibration/IDCFourierTransform.h"
#include "TRandom.h"
#include <numeric>

namespace o2::tpc
{

static constexpr float ABSTOLERANCE = 0.01f; // absolute tolerance is taken at small values near 0
static constexpr float TOLERANCE = 0.4f;     // difference between original 1D-IDC and 1D-IDC from fourier transform -> inverse fourier transform

o2::tpc::OneDIDC get1DIDCs(const std::vector<unsigned int>& integrationIntervals)
{
  const unsigned int nIDCs = std::accumulate(integrationIntervals.begin(), integrationIntervals.end(), static_cast<unsigned int>(0));
  o2::tpc::OneDIDC idcsOut;
  for (unsigned int iside = 0; iside < 2; ++iside) {
    std::vector<float> idcs(nIDCs);
    for (auto& val : idcs) {
      val = gRandom->Gaus(0, 0.2);
    }
    idcsOut.mOneDIDC[iside] = std::move(idcs);
  }
  return idcsOut;
}

std::vector<unsigned int> getIntegrationIntervalsPerTF(const unsigned int integrationIntervals, const unsigned int tfs)
{
  std::vector<unsigned int> intervals;
  intervals.reserve(tfs);
  for (unsigned int i = 0; i < tfs; ++i) {
    const unsigned int additionalInterval = (i % 3) ? 1 : 0; // in each integration inerval are either 10 or 11 values when having 128 orbits per TF and 12 orbits integration length
    intervals.emplace_back(integrationIntervals + additionalInterval);
  }
  return intervals;
}

BOOST_AUTO_TEST_CASE(IDCFourierTransform_test)
{
  const unsigned int integrationIntervals = 10;    // number of integration intervals for first TF
  const unsigned int tfs = 200;                    // number of aggregated TFs
  const unsigned int rangeIDC = 200;               // number of IDCs used to calculate the fourier coefficients
  const unsigned int nFourierCoeff = rangeIDC + 2; // number of fourier coefficients which will be calculated/stored needs to be the maximum value to be able to perform IFT
  gRandom->SetSeed(0);

  for (int iType = 0; iType < 2; ++iType) {
    const bool fft = iType == 0 ? false : true;
    o2::tpc::IDCFourierTransform::setFFT(fft);
    o2::tpc::IDCFourierTransform idcFourierTransform{rangeIDC, tfs, nFourierCoeff};
    const auto intervalsPerTF = getIntegrationIntervalsPerTF(integrationIntervals, tfs);
    idcFourierTransform.setIDCs(get1DIDCs(intervalsPerTF), intervalsPerTF);
    idcFourierTransform.setIDCs(get1DIDCs(intervalsPerTF), intervalsPerTF);
    idcFourierTransform.calcFourierCoefficients();

    const std::vector<unsigned int> offsetIndex = idcFourierTransform.getLastIntervals();
    for (unsigned int iSide = 0; iSide < o2::tpc::SIDES; ++iSide) {
      const o2::tpc::Side side = iSide == 0 ? Side::A : Side::C;
      const auto idcOneExpanded = idcFourierTransform.getExpandedIDCOne(side);
      const auto inverseFourier = idcFourierTransform.inverseFourierTransform(side);
      for (unsigned int interval = 0; interval < idcFourierTransform.getNIntervals(); ++interval) {
        for (unsigned int index = 0; index < rangeIDC; ++index) {
          const float origIDCOne = idcOneExpanded[index + offsetIndex[interval]];
          const float iFTIDCOne = inverseFourier[interval][index];
          if (std::fabs(origIDCOne) < ABSTOLERANCE) {
            BOOST_CHECK_SMALL(iFTIDCOne - origIDCOne, ABSTOLERANCE);
          } else {
            BOOST_CHECK_CLOSE(iFTIDCOne, origIDCOne, TOLERANCE);
          }
        }
      }
    }
  }
}

} // namespace o2::tpc
