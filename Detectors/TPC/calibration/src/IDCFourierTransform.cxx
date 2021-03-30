// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCCalibration/IDCFourierTransform.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonConstants/MathConstants.h"
#include "Framework/Logger.h"
#include "TFile.h"
#include <cmath>
// #include <fftw3.h>

#if (defined(WITH_OPENMP) || defined(_OPENMP)) && !defined(__CLING__)
#include <omp.h>
#endif

void o2::tpc::IDCFourierTransform::setIDCs(OneDIDC&& oneDIDCs, std::vector<unsigned int>&& integrationIntervalsPerTF)
{
  mOneDIDC[mBufferIndex] = std::move(oneDIDCs);
  mIntegrationIntervalsPerTF[mBufferIndex] = std::move(integrationIntervalsPerTF);
  mBufferIndex = !mBufferIndex;
}

void o2::tpc::IDCFourierTransform::setIDCs(const OneDIDC& oneDIDCs, const std::vector<unsigned int>& integrationIntervalsPerTF)
{
  mOneDIDC[mBufferIndex] = oneDIDCs;
  mIntegrationIntervalsPerTF[mBufferIndex] = integrationIntervalsPerTF;
  mBufferIndex = !mBufferIndex;
}

void o2::tpc::IDCFourierTransform::calcFourierCoefficientsNaive()
{
  if (mFourierCoefficients.getNCoefficientsPerTF() % 2) {
    LOGP(warning, "number of specified fourier coefficients is {}, but should be an even number! you can use FFTW3 method instead!", mFourierCoefficients.getNCoefficientsPerTF());
  }
  const std::vector<unsigned int> offsetIndex = getLastIntervals();
  calcFourierCoefficientsNaive(o2::tpc::Side::A, offsetIndex);
  calcFourierCoefficientsNaive(o2::tpc::Side::C, offsetIndex);
}

void o2::tpc::IDCFourierTransform::calcFourierCoefficientsFFTW3()
{
  LOGP(warning, "FFTW3 method not available yet. Using naive approach...");
  calcFourierCoefficientsNaive();
  // const std::vector<unsigned int> offsetIndex = getLastIntervals();
  // calcFourierCoefficientsFFTW3(o2::tpc::Side::A, offsetIndex);
  // calcFourierCoefficientsFFTW3(o2::tpc::Side::C, offsetIndex);
}

void o2::tpc::IDCFourierTransform::calcFourierCoefficientsNaive(const o2::tpc::Side side, const std::vector<unsigned int>& offsetIndex)
{
  // see: https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definitiona
#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int interval = 0; interval < getNIntervals(); ++interval) {
    const auto idcOneExpanded = getExpandedIDCOne(side);
    for (unsigned int coeff = 0; coeff < mFourierCoefficients.getNCoefficientsPerTF() / 2; ++coeff) {
      const unsigned int indexDataReal = mFourierCoefficients.getIndex(interval, 2 * coeff); // index for storing real fourier coefficient
      const unsigned int indexDataImag = indexDataReal + 1;                                  // index for storing complex fourier coefficient
      const float term0 = o2::constants::math::TwoPI * coeff / mRangeIDC;
      for (unsigned int index = 0; index < mRangeIDC; ++index) {
        const float term = term0 * index;
        const float idc0 = idcOneExpanded[index + offsetIndex[interval]];
        mFourierCoefficients(side, indexDataReal) += idc0 * std::cos(term);
        mFourierCoefficients(side, indexDataImag) -= idc0 * std::sin(term);
      }
    }
  }
  // normalize coefficient to number of used points
  normalizeCoefficients(side);
}

void o2::tpc::IDCFourierTransform::calcFourierCoefficientsFFTW3(const o2::tpc::Side side, const std::vector<unsigned int>& offsetIndex)
{
  //   // for FFTW and OMP see: https://stackoverflow.com/questions/15012054/fftw-plan-creation-using-openmp
  // #pragma omp parallel num_threads(sNThreads)
  //   {
  //     fftwf_plan fftwPlan = nullptr;
  //     float* val1DIDCs = getExpandedIDCOneFFTW(side);
  //     fftwf_complex* coefficients = fftwf_alloc_complex(getNMaxCoefficients());
  //
  // #pragma omp critical(make_plan)
  //     fftwPlan = fftwf_plan_dft_r2c_1d(mRangeIDC, val1DIDCs, coefficients, FFTW_ESTIMATE);
  //
  // #pragma omp for
  //     for (unsigned int interval = 0; interval < getNIntervals(); ++interval) {
  //       fftwf_execute_dft_r2c(fftwPlan, &(val1DIDCs[offsetIndex[interval]]), coefficients);
  //       std::memcpy(&(*(mFourierCoefficients.mFourierCoefficients[side].begin() + mFourierCoefficients.getIndex(interval, 0))), coefficients, mFourierCoefficients.getNCoefficientsPerTF() * sizeof(float));
  //     }
  //     // free memory
  //     fftwf_free(coefficients);
  //     fftwf_free(val1DIDCs);
  //     fftwf_destroy_plan(fftwPlan);
  //   }
  //   normalizeCoefficients(side);
}

std::vector<std::vector<float>> o2::tpc::IDCFourierTransform::inverseFourierTransformNaive(const o2::tpc::Side side) const
{
  // vector containing for each intervall the inverse fourier IDCs
  std::vector<std::vector<float>> inverse(getNIntervals());
  const float factor = o2::constants::math::TwoPI / mRangeIDC;

  // loop over all the intervals. For each interval the coefficients are calculated
  for (unsigned int interval = 0; interval < getNIntervals(); ++interval) {
    inverse[interval].resize(mRangeIDC);
    for (unsigned int index = 0; index < mRangeIDC; ++index) {
      const float term0 = factor * index;
      unsigned int coeffTmp = 0;
      int fac = 1; // if input data is real (and it is) the coefficients are mirrored https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
      for (unsigned int coeff = 0; coeff < mRangeIDC; ++coeff) {
        const unsigned int indexDataReal = mFourierCoefficients.getIndex(interval, 2 * coeffTmp); // index for storing real fourier coefficient
        const unsigned int indexDataImag = indexDataReal + 1;                                     // index for storing complex fourier coefficient
        const float term = term0 * coeff;
        inverse[interval][index] += mFourierCoefficients(side, indexDataReal) * std::cos(term) - fac * mFourierCoefficients(side, indexDataImag) * std::sin(term);
        if (coeff < getNMaxCoefficients() - 1) {
          ++coeffTmp;
        } else {
          --coeffTmp;
          fac = -1;
        };
      }
    }
  }
  return inverse;
}

std::vector<std::vector<float>> o2::tpc::IDCFourierTransform::inverseFourierTransformFFTW3(const o2::tpc::Side side) const
{
  LOGP(warning, "FFTW3 method not available yet. Using naive approach...");
  return inverseFourierTransformNaive(side);
  // // vector containing for each intervall the inverse fourier IDCs
  // std::vector<std::vector<float>> inverse(getNIntervals());
  //
  // // loop over all the intervals. For each interval the coefficients are calculated
  // // this loop and execution of FFTW is not optimized as it is used only for debugging
  // for (unsigned int interval = 0; interval < getNIntervals(); ++interval) {
  //   inverse[interval].resize(mRangeIDC);
  //   std::vector<std::array<float, 2>> val1DIDCs;
  //   val1DIDCs.reserve(mRangeIDC);
  //   for (unsigned int index = 0; index < getNMaxCoefficients(); ++index) {
  //     const unsigned int indexDataReal = mFourierCoefficients.getIndex(interval, 2 * index); // index for storing real fourier coefficient
  //     const unsigned int indexDataImag = indexDataReal + 1;                                  // index for storing complex fourier coefficient
  //     val1DIDCs.emplace_back(std::array<float, 2>{mFourierCoefficients(side, indexDataReal), mFourierCoefficients(side, indexDataImag)});
  //   }
  //   const fftwf_plan fftwPlan = fftwf_plan_dft_c2r_1d(mRangeIDC, reinterpret_cast<fftwf_complex*>(val1DIDCs.data()), inverse[interval].data(), FFTW_ESTIMATE);
  //   fftwf_execute(fftwPlan);
  //   fftwf_destroy_plan(fftwPlan);
  // }
  // return inverse;
}

void o2::tpc::IDCFourierTransform::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void o2::tpc::IDCFourierTransform::dumpToTree(const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();
  const std::vector<unsigned int> offsetIndex = getLastIntervals();
  for (unsigned int iSide = 0; iSide < o2::tpc::SIDES; ++iSide) {
    const o2::tpc::Side side = iSide == 0 ? Side::A : Side::C;
    const auto idcOneExpanded = getExpandedIDCOne(side);
    const auto inverseFourier = inverseFourierTransformNaive(side);
    const auto inverseFourierFFTW3 = inverseFourierTransformFFTW3(side);

    for (unsigned int interval = 0; interval < getNIntervals(); ++interval) {
      std::vector<float> oneDIDCInverse = inverseFourier[interval];
      std::vector<float> oneDIDCInverseFFTW3 = inverseFourierFFTW3[interval];

      // get 1D-IDC values used for calculation of the fourier coefficients
      std::vector<float> oneDIDC;
      oneDIDC.reserve(mRangeIDC);
      for (unsigned int index = 0; index < mRangeIDC; ++index) {
        oneDIDC.emplace_back(idcOneExpanded[index + offsetIndex[interval]]);
      }

      for (unsigned int coeff = 0; coeff < mFourierCoefficients.getNCoefficientsPerTF(); ++coeff) {
        float coefficient = mFourierCoefficients(side, mFourierCoefficients.getIndex(interval, coeff));

        pcstream << "tree"
                 << "side=" << iSide
                 << "interval=" << interval
                 << "icoefficient=" << coeff      // index of ith coefficient
                 << "coefficient=" << coefficient // value for ith coefficient
                 << "1DIDC.=" << oneDIDC
                 << "1DIDCiDFT.=" << oneDIDCInverse
                 << "1DIDiDFTFFTW3.=" << oneDIDCInverseFFTW3
                 << "\n";
      }
    }
  }
}

std::vector<unsigned int> o2::tpc::IDCFourierTransform::getLastIntervals() const
{
  std::vector<unsigned int> endIndex;
  endIndex.reserve(mTimeFrames);
  endIndex.emplace_back(0);
  for (unsigned int interval = 1; interval < mTimeFrames; ++interval) {
    endIndex.emplace_back(endIndex[interval - 1] + mIntegrationIntervalsPerTF[!mBufferIndex][interval]);
  }
  return endIndex;
}

std::vector<float> o2::tpc::IDCFourierTransform::getExpandedIDCOne(const o2::tpc::Side side) const
{
  std::vector<float> val1DIDCs = mOneDIDC[!mBufferIndex].mOneDIDC[side]; // just copy the elements
  if (useLastBuffer()) {
    val1DIDCs.insert(val1DIDCs.begin(), mOneDIDC[mBufferIndex].mOneDIDC[side].end() - mRangeIDC + mIntegrationIntervalsPerTF[!mBufferIndex][0], mOneDIDC[mBufferIndex].mOneDIDC[side].end());
  }
  return val1DIDCs;
}

// float* o2::tpc::IDCFourierTransform::getExpandedIDCOneFFTW(const o2::tpc::Side side) const
// {
// const unsigned int nElementsLastBuffer = useLastBuffer() ? mRangeIDC - mIntegrationIntervalsPerTF[!mBufferIndex][0] : 0;
// const unsigned int nElementsAll = mOneDIDC[!mBufferIndex].getNIDCs(side) + nElementsLastBuffer;
// float* val1DIDCs = fftwf_alloc_real(nElementsAll);
// if (useLastBuffer()) {
// std::memcpy(val1DIDCs, &(*(mOneDIDC[mBufferIndex].mOneDIDC[side].end() - nElementsLastBuffer)), nElementsLastBuffer * sizeof(float)); // copy IDCs from old buffer
// }
// std::memcpy(&val1DIDCs[nElementsLastBuffer], mOneDIDC[!mBufferIndex].mOneDIDC[side].data(), mOneDIDC[!mBufferIndex].getNIDCs(side) * sizeof(float)); // copy all IDCs from current buffer
// return val1DIDCs;
// }
