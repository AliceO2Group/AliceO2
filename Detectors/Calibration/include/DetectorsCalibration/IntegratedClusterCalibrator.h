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

/// \file IntegratedClusterCalibrator.h
/// \brief calibrator class for accumulating integrated clusters
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 21, 2023

#ifndef INTEGRATEDCLUSTERCALIBRATOR_H_
#define INTEGRATEDCLUSTERCALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

class TTree;

namespace o2
{

// see https://stackoverflow.com/questions/56483053/if-condition-checking-multiple-vector-sizes-are-equal
// comparing if all vector have the same size
template <typename T0, typename... Ts>
bool sameSize(T0 const& first, Ts const&... rest)
{
  return ((first.size() == rest.size()) && ...);
}

namespace tof
{

/// struct containing the integrated TOF currents
struct ITOFC {
  std::vector<float> mITOFCNCl;                                     ///< integrated 1D TOF cluster currents
  std::vector<float> mITOFCQ;                                       ///< integrated 1D TOF qTot currents
  long mTimeMS{};                                                   ///< start time in ms
  bool areSameSize() const { return sameSize(mITOFCNCl, mITOFCQ); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mITOFCNCl.empty(); }                ///< check if values are empty
  size_t getEntries() const { return mITOFCNCl.size(); }            ///< \return returns number of values stored
  void setStartTime(long timeMS) { mTimeMS = timeMS; }

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const ITOFC& data)
  {
    std::copy(data.mITOFCNCl.begin(), data.mITOFCNCl.end(), mITOFCNCl.begin() + posIndex);
    std::copy(data.mITOFCQ.begin(), data.mITOFCQ.end(), mITOFCQ.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mITOFCNCl.insert(mITOFCNCl.begin(), vecTmp.begin(), vecTmp.end());
    mITOFCQ.insert(mITOFCQ.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mITOFCNCl.resize(nTotal);
    mITOFCQ.resize(nTotal);
  }

  ClassDefNV(ITOFC, 2);
};
} // end namespace tof

namespace tpc
{

/// struct containing the integrated TPC currents
struct ITPCC {
  std::vector<float> mIQMaxA; ///< integrated 1D-currents for QMax A-side
  std::vector<float> mIQMaxC; ///< integrated 1D-currents for QMax C-side
  std::vector<float> mIQTotA; ///< integrated 1D-currents for QTot A-side
  std::vector<float> mIQTotC; ///< integrated 1D-currents for QTot A-side
  std::vector<float> mINClA;  ///< integrated 1D-currents for NCl A-side
  std::vector<float> mINClC;  ///< integrated 1D-currents for NCl A-side
  long mTimeMS{};             ///< start time in ms

  float compression(float value, const int nBits) const
  {
    const int shiftN = std::pow(2, nBits);
    int exp2;
    const auto mantissa = std::frexp(value, &exp2);
    const auto mantissaRounded = std::round(mantissa * shiftN) / shiftN;
    return std::ldexp(mantissaRounded, exp2);
  }

  void compress(const int nBits)
  {
    std::transform(mIQMaxA.begin(), mIQMaxA.end(), mIQMaxA.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mIQMaxC.begin(), mIQMaxC.end(), mIQMaxC.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mIQTotA.begin(), mIQTotA.end(), mIQTotA.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mIQTotC.begin(), mIQTotC.end(), mIQTotC.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mINClA.begin(), mINClA.end(), mINClA.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mINClC.begin(), mINClC.end(), mINClC.begin(), [this, nBits](float val) { return compression(val, nBits); });
  }

  bool areSameSize() const { return sameSize(mIQMaxA, mIQMaxC, mIQTotA, mIQTotC, mINClA, mINClC); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mIQMaxA.empty(); }                                                  ///< check if values are empty
  size_t getEntries() const { return mIQMaxA.size(); }                                              ///< \return returns number of values stored
  void setStartTime(long timeMS) { mTimeMS = timeMS; }

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const ITPCC& data)
  {
    std::copy(data.mIQMaxA.begin(), data.mIQMaxA.end(), mIQMaxA.begin() + posIndex);
    std::copy(data.mIQMaxC.begin(), data.mIQMaxC.end(), mIQMaxC.begin() + posIndex);
    std::copy(data.mIQTotA.begin(), data.mIQTotA.end(), mIQTotA.begin() + posIndex);
    std::copy(data.mIQTotC.begin(), data.mIQTotC.end(), mIQTotC.begin() + posIndex);
    std::copy(data.mINClA.begin(), data.mINClA.end(), mINClA.begin() + posIndex);
    std::copy(data.mINClC.begin(), data.mINClC.end(), mINClC.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mIQMaxA.insert(mIQMaxA.begin(), vecTmp.begin(), vecTmp.end());
    mIQMaxC.insert(mIQMaxC.begin(), vecTmp.begin(), vecTmp.end());
    mIQTotA.insert(mIQTotA.begin(), vecTmp.begin(), vecTmp.end());
    mIQTotC.insert(mIQTotC.begin(), vecTmp.begin(), vecTmp.end());
    mINClA.insert(mINClA.begin(), vecTmp.begin(), vecTmp.end());
    mINClC.insert(mINClC.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mIQMaxA.resize(nTotal);
    mIQMaxC.resize(nTotal);
    mIQTotA.resize(nTotal);
    mIQTotC.resize(nTotal);
    mINClA.resize(nTotal);
    mINClC.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mIQMaxA.begin(), mIQMaxA.end(), 0);
    std::fill(mIQMaxC.begin(), mIQMaxC.end(), 0);
    std::fill(mIQTotA.begin(), mIQTotA.end(), 0);
    std::fill(mIQTotC.begin(), mIQTotC.end(), 0);
    std::fill(mINClA.begin(), mINClA.end(), 0);
    std::fill(mINClC.begin(), mINClC.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mIQMaxA.begin(), mIQMaxA.end(), mIQMaxA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIQMaxC.begin(), mIQMaxC.end(), mIQMaxC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIQTotA.begin(), mIQTotA.end(), mIQTotA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIQTotC.begin(), mIQTotC.end(), mIQTotC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINClA.begin(), mINClA.end(), mINClA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINClC.begin(), mINClC.end(), mINClC.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(ITPCC, 2);
};

/// struct containing time series values
struct TimeSeries {
  std::vector<float> mDCAr_A_Median;       ///< integrated 1D DCAr for A-side median in phi/tgl slices
  std::vector<float> mDCAr_C_Median;       ///< integrated 1D DCAr for C-side weighted mean in phi/tgl slices
  std::vector<float> mDCAr_A_WeightedMean; ///< integrated 1D DCAr for A-side weighted mean in phi/tgl slices
  std::vector<float> mDCAr_C_WeightedMean; ///< integrated 1D DCAr for C-side median in phi/tgl slices
  std::vector<float> mDCAr_A_RMS;          ///< integrated 1D DCAr for A-side RMS in phi/tgl slices
  std::vector<float> mDCAr_C_RMS;          ///< integrated 1D DCAr for C-side RMS in phi/tgl slices
  std::vector<float> mDCAr_A_NTracks;      ///< number of tracks used to calculate the DCAs
  std::vector<float> mDCAr_C_NTracks;      ///< number of tracks used to calculate the DCAs
  std::vector<float> mDCAz_A_Median;       ///< integrated 1D DCAz for A-side median in phi/tgl slices
  std::vector<float> mDCAz_C_Median;       ///< integrated 1D DCAz for C-side median in phi/tgl slices
  std::vector<float> mDCAz_A_WeightedMean; ///< integrated 1D DCAz for A-side weighted mean in phi/tgl slices
  std::vector<float> mDCAz_C_WeightedMean; ///< integrated 1D DCAz for C-side weighted mean in phi/tgl slices
  std::vector<float> mDCAz_A_RMS;          ///< integrated 1D DCAz for A-side RMS in phi/tgl slices
  std::vector<float> mDCAz_C_RMS;          ///< integrated 1D DCAz for C-side RMS in phi/tgl slices
  std::vector<float> mDCAz_A_NTracks;      ///< number of tracks used to calculate the DCAs
  std::vector<float> mDCAz_C_NTracks;      ///< number of tracks used to calculate the DCAs
  std::vector<float> mMIPdEdxRatioA;       ///< ratio of MIP/dEdx
  std::vector<float> mMIPdEdxRatioC;       ///< ratio of MIP/dEdx
  std::vector<float> mTPCChi2A;            ///< Chi2 of TPC tracks
  std::vector<float> mTPCChi2C;            ///< Chi2 of TPC tracks
  std::vector<float> mTPCNClA;             ///< number of TPC cluster
  std::vector<float> mTPCNClC;             ///< number of TPC cluster

  unsigned char mNBinsPhi{}; ///< number of tgl bins
  unsigned char mNBinsTgl{}; ///< number of phi bins
  unsigned char mNBinsqPt{}; ///< number of qPt bins
  long mTimeMS{};            ///< start time in ms

  /// dump object to tree
  /// \param outFileName name of the output file
  /// \param nHBFPerTF number of orbits per TF
  void dumpToTree(TTree& tree, const char* prefix = "", const int nHBFPerTF = 32) const;
  void setStartTime(long timeMS) { mTimeMS = timeMS; }
  bool areSameSize() const { return sameSize(mDCAr_A_WeightedMean, mDCAr_C_WeightedMean, mDCAz_A_WeightedMean, mDCAz_C_WeightedMean, mDCAr_A_Median, mDCAr_C_Median, mDCAr_A_RMS, mDCAr_C_RMS, mDCAz_A_Median, mDCAz_C_Median, mDCAz_A_RMS, mDCAz_C_RMS, mDCAz_C_NTracks, mDCAr_A_NTracks, mDCAz_A_NTracks, mDCAz_C_NTracks); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mDCAr_A_Median.empty(); }                                                                                                                                                                                                                                                                       ///< check if values are empty

  size_t getEntries() const { return mDCAr_A_Median.size(); } ///< \return returns number of values stored
  /// \return returns total number of bins
  int getNBins() const { return mNBinsTgl + mNBinsPhi + mNBinsqPt + 1; }

  /// \return returns index for given phi bin
  int getIndexPhi(const int iPhi, int slice = 0) const { return iPhi + slice * getNBins(); }

  /// \return returns index for given tgl bin
  int getIndexTgl(const int iTgl, int slice = 0) const { return mNBinsPhi + iTgl + slice * getNBins(); }

  /// \return returns index for given qPt bin
  int getIndexqPt(const int iqPt, int slice = 0) const { return mNBinsPhi + mNBinsTgl + iqPt + slice * getNBins(); }

  /// \return returns index for integrated over all bins
  int getIndexInt(int slice = 0) const { return getNBins() - 1 + slice * getNBins(); }

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const TimeSeries& data)
  {
    mNBinsPhi = data.mNBinsPhi;
    mNBinsTgl = data.mNBinsTgl;
    mNBinsqPt = data.mNBinsqPt;
    fill(data.mDCAr_A_Median, mDCAr_A_Median, posIndex);
    fill(data.mDCAr_C_Median, mDCAr_C_Median, posIndex);
    fill(data.mDCAr_A_RMS, mDCAr_A_RMS, posIndex);
    fill(data.mDCAr_C_RMS, mDCAr_C_RMS, posIndex);
    fill(data.mDCAz_A_Median, mDCAz_A_Median, posIndex);
    fill(data.mDCAz_C_Median, mDCAz_C_Median, posIndex);
    fill(data.mDCAz_A_RMS, mDCAz_A_RMS, posIndex);
    fill(data.mDCAz_C_RMS, mDCAz_C_RMS, posIndex);
    fill(data.mDCAr_C_NTracks, mDCAr_C_NTracks, posIndex);
    fill(data.mDCAr_A_NTracks, mDCAr_A_NTracks, posIndex);
    fill(data.mDCAz_A_NTracks, mDCAz_A_NTracks, posIndex);
    fill(data.mDCAz_C_NTracks, mDCAz_C_NTracks, posIndex);
    fill(data.mDCAr_A_WeightedMean, mDCAr_A_WeightedMean, posIndex);
    fill(data.mDCAr_C_WeightedMean, mDCAr_C_WeightedMean, posIndex);
    fill(data.mDCAz_A_WeightedMean, mDCAz_A_WeightedMean, posIndex);
    fill(data.mDCAz_C_WeightedMean, mDCAz_C_WeightedMean, posIndex);
    fill(data.mMIPdEdxRatioA, mMIPdEdxRatioA, posIndex);
    fill(data.mMIPdEdxRatioC, mMIPdEdxRatioC, posIndex);
    fill(data.mTPCChi2A, mTPCChi2A, posIndex);
    fill(data.mTPCChi2C, mTPCChi2C, posIndex);
    fill(data.mTPCNClA, mTPCNClA, posIndex);
    fill(data.mTPCNClC, mTPCNClC, posIndex);
  }

  void fill(const std::vector<float>& vecFrom, std::vector<float>& vecTo, const unsigned int posIndex) { std::copy(vecFrom.begin(), vecFrom.end(), vecTo.begin() + posIndex); }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    insert(mDCAr_A_Median, vecTmp);
    insert(mDCAr_C_Median, vecTmp);
    insert(mDCAr_A_RMS, vecTmp);
    insert(mDCAr_C_RMS, vecTmp);
    insert(mDCAz_A_Median, vecTmp);
    insert(mDCAz_C_Median, vecTmp);
    insert(mDCAz_A_RMS, vecTmp);
    insert(mDCAz_C_RMS, vecTmp);
    insert(mDCAr_C_NTracks, vecTmp);
    insert(mDCAr_A_NTracks, vecTmp);
    insert(mDCAz_A_NTracks, vecTmp);
    insert(mDCAz_C_NTracks, vecTmp);
    insert(mDCAr_A_WeightedMean, vecTmp);
    insert(mDCAr_C_WeightedMean, vecTmp);
    insert(mDCAz_A_WeightedMean, vecTmp);
    insert(mDCAz_C_WeightedMean, vecTmp);
    insert(mMIPdEdxRatioA, vecTmp);
    insert(mMIPdEdxRatioC, vecTmp);
    insert(mTPCChi2A, vecTmp);
    insert(mTPCChi2C, vecTmp);
    insert(mTPCNClA, vecTmp);
    insert(mTPCNClC, vecTmp);
  }

  void insert(std::vector<float>& vec, const std::vector<float>& vecTmp) { vec.insert(vec.begin(), vecTmp.begin(), vecTmp.end()); }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mDCAr_A_Median.resize(nTotal);
    mDCAr_C_Median.resize(nTotal);
    mDCAr_A_RMS.resize(nTotal);
    mDCAr_C_RMS.resize(nTotal);
    mDCAz_A_Median.resize(nTotal);
    mDCAz_C_Median.resize(nTotal);
    mDCAz_A_RMS.resize(nTotal);
    mDCAz_C_RMS.resize(nTotal);
    mDCAr_C_NTracks.resize(nTotal);
    mDCAr_A_NTracks.resize(nTotal);
    mDCAz_A_NTracks.resize(nTotal);
    mDCAz_C_NTracks.resize(nTotal);
    mDCAr_A_WeightedMean.resize(nTotal);
    mDCAr_C_WeightedMean.resize(nTotal);
    mDCAz_A_WeightedMean.resize(nTotal);
    mDCAz_C_WeightedMean.resize(nTotal);
    mMIPdEdxRatioA.resize(nTotal);
    mMIPdEdxRatioC.resize(nTotal);
    mTPCChi2A.resize(nTotal);
    mTPCChi2C.resize(nTotal);
    mTPCNClA.resize(nTotal);
    mTPCNClC.resize(nTotal);
  }

  ClassDefNV(TimeSeries, 1);
};

struct ITSTPC_Matching {
  std::vector<float> mITSTPC_A_MatchEff;  ///< matching efficiency of ITS-TPC tracks A-side
  std::vector<float> mITSTPC_C_MatchEff;  ///< matching efficiency of ITS-TPC tracks C-side
  std::vector<float> mITSTPC_A_Chi2Match; ///< ITS-TPC chi2 A-side
  std::vector<float> mITSTPC_C_Chi2Match; ///< ITS-TPC chi2 C-side

  void dumpToTree(TTree& tree, const char* prefix, const int nHBFPerTF, const TimeSeries& timeSeriesRef) const;

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const ITSTPC_Matching& data)
  {
    std::copy(data.mITSTPC_A_MatchEff.begin(), data.mITSTPC_A_MatchEff.end(), mITSTPC_A_MatchEff.begin() + posIndex);
    std::copy(data.mITSTPC_C_MatchEff.begin(), data.mITSTPC_C_MatchEff.end(), mITSTPC_C_MatchEff.begin() + posIndex);
    std::copy(data.mITSTPC_A_Chi2Match.begin(), data.mITSTPC_A_Chi2Match.end(), mITSTPC_A_Chi2Match.begin() + posIndex);
    std::copy(data.mITSTPC_C_Chi2Match.begin(), data.mITSTPC_C_Chi2Match.end(), mITSTPC_C_Chi2Match.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mITSTPC_A_MatchEff.insert(mITSTPC_A_MatchEff.begin(), vecTmp.begin(), vecTmp.end());
    mITSTPC_C_MatchEff.insert(mITSTPC_C_MatchEff.begin(), vecTmp.begin(), vecTmp.end());
    mITSTPC_A_Chi2Match.insert(mITSTPC_A_Chi2Match.begin(), vecTmp.begin(), vecTmp.end());
    mITSTPC_C_Chi2Match.insert(mITSTPC_C_Chi2Match.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mITSTPC_A_MatchEff.resize(nTotal);
    mITSTPC_C_MatchEff.resize(nTotal);
    mITSTPC_A_Chi2Match.resize(nTotal);
    mITSTPC_C_Chi2Match.resize(nTotal);
  }

  ClassDefNV(ITSTPC_Matching, 1);
};

struct TimeSeriesITSTPC {
  TimeSeries mTSTPC;                             ///< TPC standalone DCAs
  TimeSeries mTSITSTPC;                          ///< ITS-TPC standalone DCAs
  ITSTPC_Matching mITSTPCAll;                    ///< ITS-TPC matching efficiency for ITS standalone + afterburner
  ITSTPC_Matching mITSTPCStandalone;             ///< ITS-TPC matching efficiency for ITS standalone
  ITSTPC_Matching mITSTPCAfterburner;            ///< ITS-TPC matchin efficiency  fir ITS afterburner
  std::vector<float> nPrimVertices;              ///< number of primary vertices
  std::vector<float> nVertexContributors_Median; ///< number of primary vertices
  std::vector<float> nVertexContributors_RMS;    ///< number of primary vertices
  std::vector<float> vertexX_Median;             ///< vertex x position
  std::vector<float> vertexY_Median;             ///< vertex y position
  std::vector<float> vertexZ_Median;             ///< vertex z position
  std::vector<float> vertexX_RMS;                ///< vertex x RMS
  std::vector<float> vertexY_RMS;                ///< vertex y RMS
  std::vector<float> vertexZ_RMS;                ///< vertex z RMS
  std::vector<float> mDCAr_comb_A_Median;        ///< DCAr for ITS-TPC track - A-side
  std::vector<float> mDCAz_comb_A_Median;        ///< DCAz for ITS-TPC track - A-side
  std::vector<float> mDCAr_comb_A_RMS;           ///< DCAr RMS for ITS-TPC track - A-side
  std::vector<float> mDCAz_comb_A_RMS;           ///< DCAz RMS for ITS-TPC track - A-side
  std::vector<float> mDCAr_comb_C_Median;        ///< DCAr for ITS-TPC track - C-side
  std::vector<float> mDCAz_comb_C_Median;        ///< DCAz for ITS-TPC track - C-side
  std::vector<float> mDCAr_comb_C_RMS;           ///< DCAr RMS for ITS-TPC track - C-side
  std::vector<float> mDCAz_comb_C_RMS;           ///< DCAz RMS for ITS-TPC track - C-side

  // dump this object to a tree
  void dumpToTree(const char* outFileName, const int nHBFPerTF = 32);

  void setStartTime(long timeMS)
  {
    mTSTPC.setStartTime(timeMS);
    mTSITSTPC.setStartTime(timeMS);
  }

  void setBinning(const int nBinsPhi, const int nBinsTgl, const int qPtBins)
  {
    mTSTPC.mNBinsPhi = nBinsPhi;
    mTSTPC.mNBinsTgl = nBinsTgl;
    mTSTPC.mNBinsqPt = qPtBins;
    mTSITSTPC.mNBinsPhi = nBinsPhi;
    mTSITSTPC.mNBinsTgl = nBinsTgl;
    mTSITSTPC.mNBinsqPt = qPtBins;
  }

  bool areSameSize() const { return (mTSTPC.areSameSize() && mTSITSTPC.areSameSize()); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mTSTPC.isEmpty(); }                                      ///< check if values are empty
  size_t getEntries() const { return mTSTPC.getEntries(); }                              ///< \return returns number of values stored
  void fill(const std::vector<float>& vecFrom, std::vector<float>& vecTo, const unsigned int posIndex) { std::copy(vecFrom.begin(), vecFrom.end(), vecTo.begin() + posIndex); }
  void insert(std::vector<float>& vec, const std::vector<float>& vecTmp) { vec.insert(vec.begin(), vecTmp.begin(), vecTmp.end()); }

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const TimeSeriesITSTPC& data)
  {
    mTSTPC.fill(posIndex, data.mTSTPC);
    mTSITSTPC.fill(posIndex, data.mTSITSTPC);
    mITSTPCAll.fill(posIndex, data.mITSTPCAll);
    mITSTPCStandalone.fill(posIndex, data.mITSTPCStandalone);
    mITSTPCAfterburner.fill(posIndex, data.mITSTPCAfterburner);
    fill(data.mDCAr_comb_A_Median, mDCAr_comb_A_Median, posIndex);
    fill(data.mDCAz_comb_A_Median, mDCAz_comb_A_Median, posIndex);
    fill(data.mDCAr_comb_A_RMS, mDCAr_comb_A_RMS, posIndex);
    fill(data.mDCAz_comb_A_RMS, mDCAz_comb_A_RMS, posIndex);
    fill(data.mDCAr_comb_C_Median, mDCAr_comb_C_Median, posIndex);
    fill(data.mDCAz_comb_C_Median, mDCAz_comb_C_Median, posIndex);
    fill(data.mDCAr_comb_C_RMS, mDCAr_comb_C_RMS, posIndex);
    fill(data.mDCAz_comb_C_RMS, mDCAz_comb_C_RMS, posIndex);

    const int iTF = posIndex / mTSTPC.getNBins();
    nPrimVertices[iTF] = data.nPrimVertices.front();
    nVertexContributors_Median[iTF] = data.nVertexContributors_Median.front();
    nVertexContributors_RMS[iTF] = data.nVertexContributors_RMS.front();
    vertexX_Median[iTF] = data.vertexX_Median.front();
    vertexY_Median[iTF] = data.vertexY_Median.front();
    vertexZ_Median[iTF] = data.vertexZ_Median.front();
    vertexX_RMS[iTF] = data.vertexX_RMS.front();
    vertexY_RMS[iTF] = data.vertexY_RMS.front();
    vertexZ_RMS[iTF] = data.vertexZ_RMS.front();
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    mTSTPC.insert(nDummyValues);
    mTSITSTPC.insert(nDummyValues);
    mITSTPCAll.insert(nDummyValues);
    mITSTPCStandalone.insert(nDummyValues);
    mITSTPCAfterburner.insert(nDummyValues);
    std::vector<float> vecTmp(nDummyValues, 0);
    insert(mDCAr_comb_A_Median, vecTmp);
    insert(mDCAz_comb_A_Median, vecTmp);
    insert(mDCAr_comb_A_RMS, vecTmp);
    insert(mDCAz_comb_A_RMS, vecTmp);
    insert(mDCAr_comb_C_Median, vecTmp);
    insert(mDCAz_comb_C_Median, vecTmp);
    insert(mDCAr_comb_C_RMS, vecTmp);
    insert(mDCAz_comb_C_RMS, vecTmp);

    const int nDummyValuesVtx = nDummyValues / mTSTPC.getNBins();
    std::vector<float> vecTmpVtx(nDummyValuesVtx, 0);
    insert(nPrimVertices, vecTmp);
    insert(nVertexContributors_Median, vecTmp);
    insert(nVertexContributors_RMS, vecTmp);
    insert(vertexX_Median, vecTmp);
    insert(vertexY_Median, vecTmp);
    insert(vertexZ_Median, vecTmp);
    insert(vertexX_RMS, vecTmp);
    insert(vertexY_RMS, vecTmp);
    insert(vertexZ_RMS, vecTmp);
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mTSTPC.resize(nTotal);
    mTSITSTPC.resize(nTotal);
    mITSTPCAll.resize(nTotal);
    mITSTPCStandalone.resize(nTotal);
    mITSTPCAfterburner.resize(nTotal);
    mDCAr_comb_A_Median.resize(nTotal);
    mDCAz_comb_A_Median.resize(nTotal);
    mDCAr_comb_A_RMS.resize(nTotal);
    mDCAz_comb_A_RMS.resize(nTotal);
    mDCAr_comb_C_Median.resize(nTotal);
    mDCAz_comb_C_Median.resize(nTotal);
    mDCAr_comb_C_RMS.resize(nTotal);
    mDCAz_comb_C_RMS.resize(nTotal);

    const int nTotalVtx = nTotal / mTSTPC.getNBins();
    nPrimVertices.resize(nTotalVtx);
    nVertexContributors_Median.resize(nTotalVtx);
    nVertexContributors_RMS.resize(nTotalVtx);
    vertexX_Median.resize(nTotalVtx);
    vertexY_Median.resize(nTotalVtx);
    vertexZ_Median.resize(nTotalVtx);
    vertexX_RMS.resize(nTotalVtx);
    vertexY_RMS.resize(nTotalVtx);
    vertexZ_RMS.resize(nTotalVtx);
  }

  ClassDefNV(TimeSeriesITSTPC, 2);
};

} // end namespace tpc

namespace fit
{

/// struct containing the integrated FT0 currents
struct IFT0C {
  std::vector<float> mINChanA;                                                        ///< integrated 1D FIT currents for NChan A
  std::vector<float> mINChanC;                                                        ///< integrated 1D FIT currents for NChan C
  std::vector<float> mIAmplA;                                                         ///< integrated 1D FIT currents for Ampl A
  std::vector<float> mIAmplC;                                                         ///< integrated 1D FIT currents for Ampl C
  long mTimeMS{};                                                                     ///< start time in ms
  bool areSameSize() const { return sameSize(mINChanA, mINChanC, mIAmplA, mIAmplC); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mINChanA.empty(); }                                   ///< check if values are empty
  size_t getEntries() const { return mINChanA.size(); }                               ///< \return returns number of values stored
  void setStartTime(long timeMS) { mTimeMS = timeMS; }

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const IFT0C& data)
  {
    std::copy(data.mINChanA.begin(), data.mINChanA.end(), mINChanA.begin() + posIndex);
    std::copy(data.mINChanC.begin(), data.mINChanC.end(), mINChanC.begin() + posIndex);
    std::copy(data.mIAmplA.begin(), data.mIAmplA.end(), mIAmplA.begin() + posIndex);
    std::copy(data.mIAmplC.begin(), data.mIAmplC.end(), mIAmplC.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mINChanA.insert(mINChanA.begin(), vecTmp.begin(), vecTmp.end());
    mINChanC.insert(mINChanC.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplA.insert(mIAmplA.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplC.insert(mIAmplC.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mINChanA.resize(nTotal);
    mINChanC.resize(nTotal);
    mIAmplA.resize(nTotal);
    mIAmplC.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mINChanA.begin(), mINChanA.end(), 0);
    std::fill(mINChanC.begin(), mINChanC.end(), 0);
    std::fill(mIAmplA.begin(), mIAmplA.end(), 0);
    std::fill(mIAmplC.begin(), mIAmplC.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mINChanA.begin(), mINChanA.end(), mINChanA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINChanC.begin(), mINChanC.end(), mINChanC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplA.begin(), mIAmplA.end(), mIAmplA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplC.begin(), mIAmplC.end(), mIAmplC.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(IFT0C, 2);
};

/// struct containing the integrated FV0 currents
struct IFV0C {
  std::vector<float> mINChanA;                                     ///< integrated 1D FIT currents for NChan A
  std::vector<float> mIAmplA;                                      ///< integrated 1D FIT currents for Ampl A
  long mTimeMS{};                                                  ///< start time in ms
  bool areSameSize() const { return sameSize(mINChanA, mIAmplA); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mINChanA.empty(); }                ///< check if values are empty
  size_t getEntries() const { return mINChanA.size(); }            ///< \return returns number of values stored
  void setStartTime(long timeMS) { mTimeMS = timeMS; }

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const IFV0C& data)
  {
    std::copy(data.mINChanA.begin(), data.mINChanA.end(), mINChanA.begin() + posIndex);
    std::copy(data.mIAmplA.begin(), data.mIAmplA.end(), mIAmplA.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mINChanA.insert(mINChanA.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplA.insert(mIAmplA.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mINChanA.resize(nTotal);
    mIAmplA.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mINChanA.begin(), mINChanA.end(), 0);
    std::fill(mIAmplA.begin(), mIAmplA.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mINChanA.begin(), mINChanA.end(), mINChanA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplA.begin(), mIAmplA.end(), mIAmplA.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(IFV0C, 2);
};

/// struct containing the integrated FDD currents
struct IFDDC {
  std::vector<float> mINChanA;                                                        ///< integrated 1D FIT currents for NChan A
  std::vector<float> mINChanC;                                                        ///< integrated 1D FIT currents for NChan C
  std::vector<float> mIAmplA;                                                         ///< integrated 1D FIT currents for Ampl A
  std::vector<float> mIAmplC;                                                         ///< integrated 1D FIT currents for Ampl C
  long mTimeMS{};                                                                     ///< start time in ms
  bool areSameSize() const { return sameSize(mINChanA, mINChanC, mIAmplA, mIAmplC); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mINChanA.empty(); }                                   ///< check if values are empty
  size_t getEntries() const { return mINChanA.size(); }                               ///< \return returns number of values stored
  void setStartTime(long timeMS) { mTimeMS = timeMS; }

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const IFDDC& data)
  {
    std::copy(data.mINChanA.begin(), data.mINChanA.end(), mINChanA.begin() + posIndex);
    std::copy(data.mINChanC.begin(), data.mINChanC.end(), mINChanC.begin() + posIndex);
    std::copy(data.mIAmplA.begin(), data.mIAmplA.end(), mIAmplA.begin() + posIndex);
    std::copy(data.mIAmplC.begin(), data.mIAmplC.end(), mIAmplC.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mINChanA.insert(mINChanA.begin(), vecTmp.begin(), vecTmp.end());
    mINChanC.insert(mINChanC.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplA.insert(mIAmplA.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplC.insert(mIAmplC.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mINChanA.resize(nTotal);
    mINChanC.resize(nTotal);
    mIAmplA.resize(nTotal);
    mIAmplC.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mINChanA.begin(), mINChanA.end(), 0);
    std::fill(mINChanC.begin(), mINChanC.end(), 0);
    std::fill(mIAmplA.begin(), mIAmplA.end(), 0);
    std::fill(mIAmplC.begin(), mIAmplC.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mINChanA.begin(), mINChanA.end(), mINChanA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINChanC.begin(), mINChanC.end(), mINChanC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplA.begin(), mIAmplA.end(), mIAmplA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplC.begin(), mIAmplC.end(), mIAmplC.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(IFDDC, 2);
};

} // end namespace fit
} // end namespace o2

namespace o2
{
namespace calibration
{

/// class for accumulating integrated currents
template <typename DataT>
class IntegratedClusters
{
 public:
  /// \constructor
  /// \param tFirst first TF of the stored currents
  /// \param tLast last TF of the stored currents
  IntegratedClusters(o2::calibration::TFType tFirst, o2::calibration::TFType tLast) : mTFFirst{tFirst}, mTFLast{tLast} {};

  /// \default constructor for ROOT I/O
  IntegratedClusters() = default;

  /// print summary informations
  void print() const { LOGP(info, "TF Range from {} to {} with {} of remaining data", mTFFirst, mTFLast, mRemainingData); }

  /// accumulate currents for given TF
  /// \param tfID TF ID of incoming data
  /// \param currentsContainer container containing the currents for given detector for one TF for number of clusters
  void fill(const o2::calibration::TFType tfID, const DataT& currentsContainer);

  /// merging TOF currents with previous interval
  void merge(const IntegratedClusters* prev);

  /// \return always return true. To specify the number of time slot intervals to wait for one should use the --max-delay option
  bool hasEnoughData() const { return (mRemainingData != -1); }

  /// \return returns accumulated currents
  const auto& getCurrents() const& { return mCurrents; }

  /// \return returns accumulated currents using move semantics
  auto getCurrents() && { return std::move(mCurrents); }

  /// \param currents currents for given detector which will be set
  void setCurrents(const DataT& currents) { mCurrents = currents; }

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IntegratedClusters.root", const char* outName = "IC") const;

  /// dump object to TTree for visualisation
  /// \param outFileName name of the output file
  void dumpToTree(const char* outFileName = "ICTree.root");

  /// setting the start time
  void setStartTime(long timeMS) { mCurrents.setStartTime(timeMS); }

 private:
  DataT mCurrents;                             ///< buffer for integrated currents
  o2::calibration::TFType mTFFirst{};          ///< first TF of currents
  o2::calibration::TFType mTFLast{};           ///< last TF of currents
  o2::calibration::TFType mRemainingData = -1; ///< counter for received data
  unsigned int mNValuesPerTF{};                ///< number of expected currents per TF (estimated from first received data)
  bool mInitialize{true};                      ///< flag if this object will be initialized when fill method is called

  /// init member when first data is received
  /// \param valuesPerTF number of expected values per TF
  void initData(const unsigned int valuesPerTF);

  ClassDefNV(IntegratedClusters, 1);
};

template <typename DataT>
class IntegratedClusterCalibrator : public o2::calibration::TimeSlotCalibration<IntegratedClusters<DataT>>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<IntegratedClusters<DataT>>;
  using CalibVector = std::vector<DataT>;
  using TFinterval = std::vector<std::pair<TFType, TFType>>;
  using TimeInterval = std::vector<std::pair<long, long>>;

 public:
  /// default constructor
  IntegratedClusterCalibrator() = default;

  /// default destructor
  ~IntegratedClusterCalibrator() final = default;

  /// check if given slot has already enough data
  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->hasEnoughData(); }

  /// clearing all calibration objects in the output buffer
  void initOutput() final;

  /// storing the integrated currents for given slot
  void finalizeSlot(Slot& slot) final;

  /// Creates new time slot
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  /// \return CCDB output informations
  const TFinterval& getTFinterval() const { return mIntervals; }

  /// \return Time frame time information
  const TimeInterval& getTimeIntervals() const { return mTimeIntervals; }

  /// \return returns calibration objects (pad-by-pad gain maps)
  auto getCalibs() && { return std::move(mCalibs); }

  /// check if calibration data is available
  bool hasCalibrationData() const { return mCalibs.size() > 0; }

  /// set if debug objects will be created
  void setDebug(const bool debug) { mDebug = debug; }

 private:
  TFinterval mIntervals;       ///< start and end time frames of each calibration time slots
  TimeInterval mTimeIntervals; ///< start and end times of each calibration time slots
  CalibVector mCalibs;         ///< Calibration object containing for each pad a histogram with normalized charge
  bool mDebug{false};          ///< write debug output objects

  ClassDefOverride(IntegratedClusterCalibrator, 1);
};

} // end namespace calibration
} // end namespace o2

#endif
