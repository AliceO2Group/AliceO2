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

#include "TPCCalibration/IDCFourierTransform.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonConstants/MathConstants.h"
#include "Framework/Logger.h"
#include "TFile.h"
#include <fftw3.h>

#if (defined(WITH_OPENMP) || defined(_OPENMP)) && !defined(__CLING__)
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
#endif

template <class Type>
o2::tpc::IDCFourierTransform<Type>::~IDCFourierTransform()
{
  for (int thread = 0; thread < sNThreads; ++thread) {
    fftwf_free(mVal1DIDCs[thread]);
    fftwf_free(mCoefficients[thread]);
  }
  fftwf_destroy_plan(mFFTWPlan);
}

template <class Type>
void o2::tpc::IDCFourierTransform<Type>::initFFTW3Members()
{
  for (int thread = 0; thread < sNThreads; ++thread) {
    mVal1DIDCs[thread] = fftwf_alloc_real(this->mRangeIDC);
    mCoefficients[thread] = fftwf_alloc_complex(getNMaxCoefficients());
  }
  mFFTWPlan = fftwf_plan_dft_r2c_1d(this->mRangeIDC, mVal1DIDCs.front(), mCoefficients.front(), FFTW_ESTIMATE);
}

template <class Type>
void o2::tpc::IDCFourierTransform<Type>::calcFourierCoefficientsNaive()
{
  LOGP(info, "calculating fourier coefficients for current TF using naive approach using {} threads", sNThreads);

  // check if IDCs are present for current side
  if (this->getNIDCs() == 0) {
    LOGP(warning, "no 1D-IDCs found!");
    mFourierCoefficients.reset();
    return;
  }

  const auto offsetIndex = this->getLastIntervals();

  // see: https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definitiona
  const bool add = mFourierCoefficients.getNCoefficientsPerTF() % 2;
  const unsigned int lastCoeff = mFourierCoefficients.getNCoefficientsPerTF() / 2;

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int interval = 0; interval < this->getNIntervals(); ++interval) {
    const std::vector<float>& idcOneExpanded{this->getExpandedIDCOne()}; // 1D-IDC values which will be used for the FFT
    for (unsigned int coeff = 0; coeff < lastCoeff; ++coeff) {
      const unsigned int indexDataReal = mFourierCoefficients.getIndex(interval, 2 * coeff); // index for storing real fourier coefficient
      const unsigned int indexDataImag = indexDataReal + 1;                                  // index for storing complex fourier coefficient
      const float term0 = o2::constants::math::TwoPI * coeff / this->mRangeIDC;
      for (unsigned int index = 0; index < this->mRangeIDC; ++index) {
        const float term = term0 * index;
        const float idc0 = idcOneExpanded[index + offsetIndex[interval]];
        mFourierCoefficients(indexDataReal) += idc0 * std::cos(term);
        mFourierCoefficients(indexDataImag) -= idc0 * std::sin(term);
      }
    }
    if (add) {
      const unsigned int indexDataReal = mFourierCoefficients.getIndex(interval, 2 * lastCoeff); // index for storing real fourier coefficient
      const float term0 = o2::constants::math::TwoPI * lastCoeff / this->mRangeIDC;
      for (unsigned int index = 0; index < this->mRangeIDC; ++index) {
        const float term = term0 * index;
        const float idc0 = idcOneExpanded[index + offsetIndex[interval]];
        mFourierCoefficients(indexDataReal) += idc0 * std::cos(term);
      }
    }
  }
  // normalize coefficient to number of used points
  normalizeCoefficients();
}

template <class Type>
void o2::tpc::IDCFourierTransform<Type>::calcFourierCoefficientsFFTW3()
{
  LOGP(info, "calculating fourier coefficients for current TF using fftw3 using {} threads", sNThreads);

  // for FFTW and OMP see: https://stackoverflow.com/questions/15012054/fftw-plan-creation-using-openmp
  // check if IDCs are present for current side
  if (this->getNIDCs() == 0) {
    LOGP(warning, "no 1D-IDCs found!");
    mFourierCoefficients.reset();
    return;
  }

  const std::vector<unsigned int> offsetIndex = this->getLastIntervals();
  const std::vector<float>& idcOneExpanded{this->getExpandedIDCOne()}; // 1D-IDC values which will be used for the FFT

  if constexpr (std::is_same_v<Type, IDCFourierTransformBaseAggregator>) {
#pragma omp parallel for num_threads(sNThreads)
    for (unsigned int interval = 0; interval < this->getNIntervals(); ++interval) {
      fftwLoop(idcOneExpanded, offsetIndex, interval, omp_get_thread_num());
    }
  } else {
    fftwLoop(idcOneExpanded, offsetIndex, 0, 0);
  }

  normalizeCoefficients();
}

template <class Type>
inline void o2::tpc::IDCFourierTransform<Type>::fftwLoop(const std::vector<float>& idcOneExpanded, const std::vector<unsigned int>& offsetIndex, const unsigned int interval, const unsigned int thread)
{
  std::memcpy(mVal1DIDCs[thread], &idcOneExpanded[offsetIndex[interval]], this->mRangeIDC * sizeof(float));                                                                                               // copy IDCs to avoid seg fault when using SIMD instructions
  fftwf_execute_dft_r2c(mFFTWPlan, mVal1DIDCs[thread], mCoefficients[thread]);                                                                                                                            // perform ft
  std::memcpy(&(*(mFourierCoefficients.mFourierCoefficients.begin() + mFourierCoefficients.getIndex(interval, 0))), mCoefficients[thread], mFourierCoefficients.getNCoefficientsPerTF() * sizeof(float)); // store coefficients
}

template <class Type>
std::vector<std::vector<float>> o2::tpc::IDCFourierTransform<Type>::inverseFourierTransformNaive() const
{
  if (this->mRangeIDC % 2) {
    LOGP(info, "number of specified fourier coefficients is {}, but should be an even number! FFTW3 method is used instead!", mFourierCoefficients.getNCoefficientsPerTF());
    return inverseFourierTransformFFTW3();
  }

  // vector containing for each intervall the inverse fourier IDCs
  std::vector<std::vector<float>> inverse(this->getNIntervals());
  const float factor = o2::constants::math::TwoPI / this->mRangeIDC;

  // loop over all the intervals. For each interval the coefficients are calculated
  for (unsigned int interval = 0; interval < this->getNIntervals(); ++interval) {
    inverse[interval].resize(this->mRangeIDC);
    for (unsigned int index = 0; index < this->mRangeIDC; ++index) {
      const float term0 = factor * index;
      unsigned int coeffTmp = 0;
      int fac = 1; // if input data is real (and it is) the coefficients are mirrored https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
      for (unsigned int coeff = 0; coeff < this->mRangeIDC; ++coeff) {
        const unsigned int indexDataReal = mFourierCoefficients.getIndex(interval, 2 * coeffTmp); // index for storing real fourier coefficient
        const unsigned int indexDataImag = indexDataReal + 1;                                     // index for storing complex fourier coefficient
        const float term = term0 * coeff;
        inverse[interval][index] += mFourierCoefficients(indexDataReal) * std::cos(term) - fac * mFourierCoefficients(indexDataImag) * std::sin(term);
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

template <class Type>
std::vector<std::vector<float>> o2::tpc::IDCFourierTransform<Type>::inverseFourierTransformFFTW3() const
{
  // vector containing for each intervall the inverse fourier IDCs
  std::vector<std::vector<float>> inverse(this->getNIntervals());

  // loop over all the intervals. For each interval the coefficients are calculated
  // this loop and execution of FFTW is not optimized as it is used only for debugging
  for (unsigned int interval = 0; interval < this->getNIntervals(); ++interval) {
    inverse[interval].resize(this->mRangeIDC);
    std::vector<std::array<float, 2>> val1DIDCs;
    val1DIDCs.reserve(this->mRangeIDC);
    for (unsigned int index = 0; index < getNMaxCoefficients(); ++index) {
      const unsigned int indexDataReal = mFourierCoefficients.getIndex(interval, 2 * index); // index for storing real fourier coefficient
      const unsigned int indexDataImag = indexDataReal + 1;                                  // index for storing complex fourier coefficient
      val1DIDCs.emplace_back(std::array<float, 2>{mFourierCoefficients(indexDataReal), mFourierCoefficients(indexDataImag)});
    }
    const fftwf_plan fftwPlan = fftwf_plan_dft_c2r_1d(this->mRangeIDC, reinterpret_cast<fftwf_complex*>(val1DIDCs.data()), inverse[interval].data(), FFTW_ESTIMATE);
    fftwf_execute(fftwPlan);
    fftwf_destroy_plan(fftwPlan);
  }
  return inverse;
}

template <class Type>
void o2::tpc::IDCFourierTransform<Type>::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

template <class Type>
void o2::tpc::IDCFourierTransform<Type>::dumpToTree(const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();
  const std::vector<unsigned int> offsetIndex = this->getLastIntervals();
  const auto idcOneExpanded = this->getExpandedIDCOne();
  const auto inverseFourier = inverseFourierTransformNaive();
  const auto inverseFourierFFTW3 = inverseFourierTransformFFTW3();

  for (unsigned int interval = 0; interval < this->getNIntervals(); ++interval) {
    std::vector<float> oneDIDCInverse = inverseFourier[interval];
    std::vector<float> oneDIDCInverseFFTW3 = inverseFourierFFTW3[interval];

    // get 1D-IDC values used for calculation of the fourier coefficients
    std::vector<float> oneDIDC;
    oneDIDC.reserve(this->mRangeIDC);
    for (unsigned int index = 0; index < this->mRangeIDC; ++index) {
      oneDIDC.emplace_back(idcOneExpanded[index + offsetIndex[interval]]);
    }

    for (unsigned int coeff = 0; coeff < mFourierCoefficients.getNCoefficientsPerTF(); ++coeff) {
      float coefficient = mFourierCoefficients(mFourierCoefficients.getIndex(interval, coeff));

      pcstream << "tree"
               << "interval=" << interval
               << "icoefficient=" << coeff      // index of ith coefficient
               << "coefficient=" << coefficient // value for ith coefficient
               << "1DIDC.=" << oneDIDC
               << "1DIDCiDFT.=" << oneDIDCInverse
               << "1DIDCiDFTFFTW3.=" << oneDIDCInverseFFTW3
               << "\n";
    }
  }
}

template <class Type>
void o2::tpc::IDCFourierTransform<Type>::printFFTWPlan() const
{
  float* val1DIDCs = fftwf_alloc_real(this->mRangeIDC);
  fftwf_complex* coefficients = fftwf_alloc_complex(getNMaxCoefficients());
  fftwf_plan fftwPlan = fftwf_plan_dft_r2c_1d(this->mRangeIDC, val1DIDCs, coefficients, FFTW_ESTIMATE);
  char* splan = fftwf_sprint_plan(fftwPlan);

  LOGP(info, "========= printing FFTW plan ========= \n {}", splan);
  double add = 0;
  double mul = 0;
  double fusedMultAdd = 0;
  fftwf_flops(fftwPlan, &add, &mul, &fusedMultAdd);
  LOGP(info, "additions: {}    multiplications: {}    fused multiply-add: {}    sum: {}", add, mul, fusedMultAdd, add + mul + fusedMultAdd);

  // free memory
  free(splan);
  fftwf_free(coefficients);
  fftwf_free(val1DIDCs);
  fftwf_destroy_plan(fftwPlan);
}

template <class Type>
std::vector<std::pair<float, float>> o2::tpc::IDCFourierTransform<Type>::getFrequencies(const o2::tpc::FourierCoeff& coeff, const float samplingFrequency)
{
  std::vector<std::pair<float, float>> freq;
  const auto nCoeff = coeff.getNCoefficientsPerTF();
  const int nFreqPerInterval = nCoeff / 2;
  const int nTFs = coeff.getNTimeFrames();
  freq.reserve(nTFs * nFreqPerInterval);
  for (int iTF = 0; iTF < nTFs; ++iTF) {
    for (int iFreq = 0; iFreq < nFreqPerInterval; ++iFreq) {
      const int realInd = nCoeff * iTF + iFreq * 2;
      const int compInd = realInd + 1;
      const float magnitude = std::sqrt(coeff(realInd) * coeff(realInd) + coeff(compInd) * coeff(compInd));
      const float freqTmp = iFreq * samplingFrequency / nCoeff;
      freq.emplace_back(freqTmp, magnitude);
    }
  }
  return freq;
}

template class o2::tpc::IDCFourierTransform<o2::tpc::IDCFourierTransformBaseEPN>;
template class o2::tpc::IDCFourierTransform<o2::tpc::IDCFourierTransformBaseAggregator>;
