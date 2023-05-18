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

/// \file IDCFourierTransform.h
/// \brief class for calculating the fourier coefficients from 1D-IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date May 11, 2021

#ifndef ALICEO2_IDCFOURIERTRANSFORM_H_
#define ALICEO2_IDCFOURIERTRANSFORM_H_

#include <vector>
#include "Rtypes.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCCalibration/IDCFourierTransformBase.h"
#include "CommonConstants/LHCConstants.h"

using fftwf_plan = struct fftwf_plan_s*;
using fftwf_complex = float[2];

namespace o2::tpc
{

/// class for fourier transform of 1D-IDCs
/// For example usage see testO2TPCIDCFourierTransform.cxx

/// \tparam Type type which can either be  IDCFourierTransformBaseEPN for synchronous reconstruction or  IDCFourierTransformBaseAggregator for aggregator
template <class Type> // do not use enum class as type to avoid problems with ROOT dictionary generation!
class IDCFourierTransform : public IDCFourierTransformBase<Type>
{
 public:
  /// constructor for  AGGREGATOR type
  /// \param rangeIDC number of IDCs for each interval which will be used to calculate the fourier coefficients
  /// \param timeFrames number of time frames which will be stored
  /// \param nFourierCoefficientsStore number of courier coefficients (real+imag) which will be stored (the maximum can be 'rangeIDC + 2', should be an even number when using naive FT). If less than maximum is setn the inverse fourier transform will not work.
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCFourierTransformBaseAggregator>::value)), int>::type = 0>
  IDCFourierTransform(const unsigned int rangeIDC = 200, const unsigned int nFourierCoefficientsStore = 200 + 2) : IDCFourierTransformAggregator(rangeIDC), mFourierCoefficients{1, nFourierCoefficientsStore}, mVal1DIDCs(sNThreads), mCoefficients(sNThreads)
  {
    initFFTW3Members();
  };

  /// constructor for  EPN type
  /// \param rangeIDC number of IDCs for each interval which will be used to calculate the fourier coefficients
  /// \param nFourierCoefficientsStore number of courier coefficients (real+imag) which will be stored (the maximum can be 'rangeIDC + 2', should be an even number when using naive FT). If less than maximum is setn the inverse fourier transform will not work.
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCFourierTransformBaseEPN>::value)), int>::type = 0>
  IDCFourierTransform(const unsigned int rangeIDC = 200, const unsigned int nFourierCoefficientsStore = 200 + 2) : IDCFourierTransformEPN(rangeIDC), mFourierCoefficients{1, nFourierCoefficientsStore}, mVal1DIDCs(sNThreads), mCoefficients(sNThreads)
  {
    initFFTW3Members();
  };

  // Destructor
  ~IDCFourierTransform();

  /// set fast fourier transform using FFTW3
  /// \param fft use FFTW3 or not (naive approach)
  static void setFFT(const bool fft) { sFftw = fft; }

  /// This function has to be called before the constructor is called
  /// \param nThreads set the number of threads used for calculation of the fourier coefficients
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCFourierTransformBaseAggregator>::value)), int>::type = 0>
  static void setNThreads(const int nThreads)
  {
    sNThreads = nThreads;
  }

  /// calculate fourier coefficients for one TPC side
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCFourierTransformBaseAggregator>::value)), int>::type = 0>
  void calcFourierCoefficients(const unsigned int timeFrames = 2000)
  {
    mFourierCoefficients.resize(timeFrames);
    sFftw ? calcFourierCoefficientsFFTW3() : calcFourierCoefficientsNaive();
  }

  /// calculate fourier coefficients for one TPC side
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCFourierTransformBaseEPN>::value)), int>::type = 0>
  void calcFourierCoefficients()
  {
    sFftw ? calcFourierCoefficientsFFTW3() : calcFourierCoefficientsNaive();
  }

  /// get IDC0 values from the inverse fourier transform. Can be used for debugging. std::vector<std::vector<float>>: first vector interval second vector IDC0 values
  std::vector<std::vector<float>> inverseFourierTransform() const { return sFftw ? inverseFourierTransformFFTW3() : inverseFourierTransformNaive(); }

  /// \return returns number of IDCs for each interval which will be used to calculate the fourier coefficients
  unsigned int getrangeIDC() const { return this->mRangeIDC; }

  /// \return returns struct holding all fourier coefficients
  const auto& getFourierCoefficients() const { return mFourierCoefficients; }

  /// get type of used fourier transform
  static bool getFFT() { return sFftw; }

  /// get the number of threads used for calculation of the fourier coefficients
  static int getNThreads() { return sNThreads; }

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "Fourier.root", const char* outName = "FourierCoefficients") const;

  /// create debug tree
  /// \param outFileName name of the output tree
  void dumpToTree(const char* outFileName = "FourierTree.root") const;

  /// printing information about the algorithms which are used by FFTW for debugging e.g. seeing if SIMD instructions will be used
  void printFFTWPlan() const;

  /// return the frequencies and the magnitude of the frequency. std::pair of <frequency, magnitude>
  /// \param samplingFrequency sampling frequency of the signal in Hz (default is IDC sampling rate in Hz)
  std::vector<std::pair<float, float>> getFrequencies(const float samplingFrequency = getSamplingFrequencyIDCHz()) const { return getFrequencies(mFourierCoefficients, samplingFrequency); }

  /// return the frequencies and the magnitude of the frequency
  /// \param coeff fourier coefficients
  /// \param samplingFrequency sampling frequency of the signal in Hz (default is IDC sampling rate in Hz)
  static std::vector<std::pair<float, float>> getFrequencies(const FourierCoeff& coeff, const float samplingFrequency = getSamplingFrequencyIDCHz());

  /// \return returns sampling frequency of IDCs in Hz
  static float getSamplingFrequencyIDCHz() { return 1e6 / (12 * o2::constants::lhc::LHCOrbitMUS); }

 private:
  FourierCoeff mFourierCoefficients;         ///< fourier coefficients. interval -> coefficient
  inline static int sFftw{1};                ///< using fftw or naive approach for calculation of fourier coefficients
  inline static int sNThreads{1};            ///< number of threads which are used during the calculation of the fourier coefficients
  fftwf_plan mFFTWPlan{nullptr};             ///<! FFTW plan which is used during the ft
  std::vector<float*> mVal1DIDCs;            ///<! buffer for the 1D-IDC values for SIMD usage (each thread will get his one obejct)
  std::vector<fftwf_complex*> mCoefficients; ///<! buffer for coefficients (each thread will get his one obejct)

  /// calculate fourier coefficients
  void calcFourierCoefficientsNaive();

  /// calculate fourier coefficients
  void calcFourierCoefficientsFFTW3();

  /// get IDC0 values from the inverse fourier transform. Can be used for debugging. std::vector<std::vector<float>>: first vector interval second vector IDC0 values
  std::vector<std::vector<float>> inverseFourierTransformNaive() const;

  /// get IDC0 values from the inverse fourier transform using FFTW3. Can be used for debugging. std::vector<std::vector<float>>: first vector interval second vector IDC0 values
  std::vector<std::vector<float>> inverseFourierTransformFFTW3() const;

  /// divide coefficients by number of IDCs used
  void normalizeCoefficients()
  {
    std::transform(mFourierCoefficients.mFourierCoefficients.begin(), mFourierCoefficients.mFourierCoefficients.end(), mFourierCoefficients.mFourierCoefficients.begin(), [norm = this->mRangeIDC](auto& val) { return val / norm; });
  };

  /// \return returns maximum numbers of stored real/imag fourier coeffiecients
  unsigned int getNMaxCoefficients() const { return this->mRangeIDC / 2 + 1; }

  /// initalizing fftw members
  void initFFTW3Members();

  /// performing of ft using FFTW
  void fftwLoop(const std::vector<float>& idcOneExpanded, const std::vector<unsigned int>& offsetIndex, const unsigned int interval, const unsigned int thread);

  ClassDefNV(IDCFourierTransform, 1)
};

} // namespace o2::tpc

#endif
