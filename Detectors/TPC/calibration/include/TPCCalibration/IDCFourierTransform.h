// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2::tpc
{

/// class for fourier transform of 1D-IDCs
/// For example usage see testO2TPCIDCFourierTransform.cxx

class IDCFourierTransform
{
 public:
  /// contructor
  /// \param rangeIDC number of IDCs for each interval which will be used to calculate the fourier coefficients
  /// \param timeFrames number of time frames which will be stored
  /// \param nFourierCoefficientsStore number of courier coefficients (real+imag) which will be stored (the maximum can be 'rangeIDC + 2', should be an even number when using naive FT). If less than maximum is setn the inverse fourier transform will not work.
  IDCFourierTransform(const unsigned int rangeIDC = 200, const unsigned int timeFrames = 2000, const unsigned int nFourierCoefficientsStore = 200 + 2) : mRangeIDC{rangeIDC}, mTimeFrames{timeFrames}, mFourierCoefficients{mTimeFrames, nFourierCoefficientsStore} {};

  /// set input 1D-IDCs which are used to calculate fourier coefficients
  /// \param oneDIDCs 1D-IDCs
  /// \param integrationIntervalsPerTF vector containg for each TF the number of IDCs
  void setIDCs(OneDIDC&& oneDIDCs, std::vector<unsigned int>&& integrationIntervalsPerTF);

  /// set input 1D-IDCs which are used to calculate fourier coefficients
  /// \param oneDIDCs 1D-IDCs
  /// \param integrationIntervalsPerTF vector containg for each TF the number of IDCs
  void setIDCs(const OneDIDC& oneDIDCs, const std::vector<unsigned int>& integrationIntervalsPerTF);

  /// set fast fourier transform using FFTW3
  /// \param fft use FFTW3 or not (naive approach)
  static void setFFT(const bool fft) { sFftw = fft; }

  /// \param nThreads set the number of threads used for calculation of the fourier coefficients
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// calculate fourier coefficients
  void calcFourierCoefficients() { sFftw ? calcFourierCoefficientsFFTW3() : calcFourierCoefficientsNaive(); }

  /// get IDC0 values from the inverse fourier transform. Can be used for debugging. std::vector<std::vector<float>>: first vector interval second vector IDC0 values
  /// \param side TPC side
  std::vector<std::vector<float>> inverseFourierTransform(const o2::tpc::Side side) const { return sFftw ? inverseFourierTransformFFTW3(side) : inverseFourierTransformNaive(side); }

  /// \return returns number of IDCs for each interval which will be used to calculate the fourier coefficients
  unsigned int getrangeIDC() const { return mRangeIDC; }

  /// \return returns number of 1D-IDCs
  /// \param side TPC side
  unsigned long getNIDCs(const o2::tpc::Side side) const { return mOneDIDC[!mBufferIndex].mOneDIDC[side].size(); }

  /// \return returns number of time frames for which the coefficients are obtained
  unsigned int getNIntervals() const { return mTimeFrames; }

  /// \return returns struct holding all fourier coefficients
  const auto& getFourierCoefficients() const { return mFourierCoefficients; }

  /// \return returns struct of stored 1D-IDC
  const OneDIDC& getOneDIDC() const { return mOneDIDC[!mBufferIndex]; }

  /// \return returns indices used for accessing correct IDCs for given TF
  std::vector<unsigned int> getLastIntervals() const;

  /// copy over IDCs from buffer to current IDCOne vector for easier access
  /// \return returns expanded 1D-IDC vector
  /// \param side TPC side
  std::vector<float> getExpandedIDCOne(const o2::tpc::Side side) const;

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

 private:
  const unsigned int mRangeIDC{};                                          ///< number of IDCs used for the calculation of fourier coefficients
  const unsigned int mTimeFrames{};                                        ///< number of timeframes which for which teh fourier coefficients are stored
  FourierCoeff mFourierCoefficients;                                       ///< fourier coefficients. side -> interval -> coefficient
  std::array<OneDIDC, 2> mOneDIDC{OneDIDC(mRangeIDC), OneDIDC(mRangeIDC)}; ///< all 1D-IDCs which are used to calculate the fourier coefficients. A buffer for the last aggregation interval is used to calculate the fourier coefficients for the first TFs
  std::array<std::vector<unsigned int>, 2> mIntegrationIntervalsPerTF{};   ///< number of integration intervals per TF used to set the correct range of IDCs. A buffer is needed for the last aggregation interval.
  bool mBufferIndex{true};                                                 ///< index for the buffer
  inline static int sFftw{1};                                              ///< using fftw or naive approach for calculation of fourier coefficients
  inline static int sNThreads{1};                                          ///< number of threads which are used during the calculation of the fourier coefficients

  /// calculate fourier coefficients
  void calcFourierCoefficientsNaive();

  /// calculate fourier coefficients
  /// \param side TPC side
  /// \param offsetIndex for accessing index obtained from getLastIntervals()
  void calcFourierCoefficientsNaive(const o2::tpc::Side side, const std::vector<unsigned int>& offsetIndex);

  /// calculate fourier coefficients
  void calcFourierCoefficientsFFTW3();

  /// calculate fourier coefficients using FFTW3 package
  /// get IDC0 values from the inverse fourier transform. Can be used for debugging. std::vector<std::vector<float>>: first vector interval second vector IDC0 values
  /// \param side TPC side
  /// \param offsetIndex for accessing index obtained from getLastIntervals()
  void calcFourierCoefficientsFFTW3(const o2::tpc::Side side, const std::vector<unsigned int>& offsetIndex);

  /// get IDC0 values from the inverse fourier transform. Can be used for debugging. std::vector<std::vector<float>>: first vector interval second vector IDC0 values
  /// \param side TPC side
  std::vector<std::vector<float>> inverseFourierTransformNaive(const o2::tpc::Side side) const;

  /// get IDC0 values from the inverse fourier transform using FFTW3. Can be used for debugging. std::vector<std::vector<float>>: first vector interval second vector IDC0 values
  /// \param side TPC side
  std::vector<std::vector<float>> inverseFourierTransformFFTW3(const o2::tpc::Side side) const;

  /// copy over IDCs from buffer to current IDCOne vector for easier access using fftwf_alloc_real for possibly/forcing SIMD (?) http://www.fftw.org/fftw3_doc/SIMD-alignment-and-fftw_005fmalloc.html
  /// \param side TPC side
  /// \param val1DIDCs 1D-IDCs which are allocated using fftwf_alloc_real
  float* getExpandedIDCOneFFTW(const o2::tpc::Side side) const;

  /// divide coefficients by number of IDCs used
  void normalizeCoefficients(const o2::tpc::Side side)
  {
    std::transform(mFourierCoefficients.mFourierCoefficients[side].begin(), mFourierCoefficients.mFourierCoefficients[side].end(), mFourierCoefficients.mFourierCoefficients[side].begin(), [norm = mRangeIDC](auto& val) { return val / norm; });
  };

  /// returns whether the buffer has to be used
  bool useLastBuffer() const { return (mRangeIDC > mIntegrationIntervalsPerTF[!mBufferIndex][0]); }

  /// \return returns maximum numbers of stored real/imag fourier coeffiecients
  unsigned int getNMaxCoefficients() const { return mRangeIDC / 2 + 1; }

  ClassDefNV(IDCFourierTransform, 1)
};

} // namespace o2::tpc

#endif
