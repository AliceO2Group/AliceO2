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

/// \file GPUQA.h
/// \author David Rohr

#ifndef GPUQA_H
#define GPUQA_H

#include "GPUSettings.h"
struct AliHLTTPCClusterMCWeight;
class TH1F;
class TH2F;
class TCanvas;
class TPad;
class TLegend;
class TPad;
class TH1;
class TFile;
class TH1D;
class TObjArray;
class TColor;
class TGraphAsymmErrors;
typedef int16_t Color_t;

#if !defined(GPUCA_BUILD_QA) || defined(GPUCA_GPUCODE)

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUQA
{
 public:
  GPUQA(void* chain) {}
  ~GPUQA() = default;
  typedef int32_t mcLabelI_t;
  int32_t InitQA(int32_t tasks = 0) { return 1; }
  void RunQA(bool matchOnly = false) {}
  int32_t DrawQAHistograms() { return 1; }
  void SetMCTrackRange(int32_t min, int32_t max) {}
  bool SuppressTrack(int32_t iTrack) const { return false; }
  bool SuppressHit(int32_t iHit) const { return false; }
  int32_t HitAttachStatus(int32_t iHit) const { return false; }
  mcLabelI_t GetMCTrackLabel(uint32_t trackId) const { return -1; }
  bool clusterRemovable(int32_t attach, bool prot) const { return false; }
  void DumpO2MCData(const char* filename) const {}
  int32_t ReadO2MCData(const char* filename) { return 1; }
  void* AllocateScratchBuffer(size_t nBytes) { return nullptr; }
  static bool QAAvailable() { return false; }
  static bool IsInitialized() { return false; }
  void UpdateChain(GPUChainTracking* chain) {}
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else

#include "GPUTPCDef.h"
#include <cmath>
#include <vector>
#include <memory>
#ifdef GPUCA_TPC_GEOMETRY_O2
#include <gsl/span>
#endif

namespace o2
{
class MCCompLabel;
namespace tpc
{
class TrackTPC;
struct ClusterNativeAccess;
} // namespace tpc
} // namespace o2

struct AliHLTTPCClusterMCLabel;

namespace GPUCA_NAMESPACE::gpu
{
class GPUChainTracking;
struct GPUParam;
struct GPUTPCMCInfo;
struct GPUQAGarbageCollection;

class GPUQA
{
 public:
  GPUQA();
  GPUQA(GPUChainTracking* chain, const GPUSettingsQA* config = nullptr, const GPUParam* param = nullptr);
  ~GPUQA();

#ifdef GPUCA_TPC_GEOMETRY_O2
  using mcLabels_t = gsl::span<const o2::MCCompLabel>;
  using mcLabel_t = o2::MCCompLabel;
  using mcLabelI_t = mcLabel_t;
#else
  using mcLabels_t = AliHLTTPCClusterMCLabel;
  using mcLabel_t = AliHLTTPCClusterMCWeight;

 private:
  struct mcLabelI_t;

 public:
#endif

  void UpdateParam(const GPUParam* param) { mParam = param; }
  int32_t InitQA(int32_t tasks = -1);
  void RunQA(bool matchOnly = false, const std::vector<o2::tpc::TrackTPC>* tracksExternal = nullptr, const std::vector<o2::MCCompLabel>* tracksExtMC = nullptr, const o2::tpc::ClusterNativeAccess* clNative = nullptr);
  int32_t DrawQAHistograms(TObjArray* qcout = nullptr);
  void DrawQAHistogramsCleanup(); // Needed after call to DrawQAHistograms with qcout != nullptr when GPUSettingsQA.shipToQCAsCanvas = true to clean up the Canvases etc.
  void SetMCTrackRange(int32_t min, int32_t max);
  bool SuppressTrack(int32_t iTrack) const;
  bool SuppressHit(int32_t iHit) const;
  int32_t HitAttachStatus(int32_t iHit) const;
  mcLabelI_t GetMCTrackLabel(uint32_t trackId) const;
  uint32_t GetMCLabelCol(const mcLabel_t& label) const;
  bool clusterRemovable(int32_t attach, bool prot) const;
  void InitO2MCData(GPUTrackingInOutPointers* updateIOPtr = nullptr);
  void DumpO2MCData(const char* filename) const;
  int32_t ReadO2MCData(const char* filename);
  static bool QAAvailable() { return true; }
  bool IsInitialized() { return mQAInitialized; }
  void UpdateChain(GPUChainTracking* chain) { mTracking = chain; }

  const std::vector<TH1F>& getHistograms1D() const { return *mHist1D; }
  const std::vector<TH2F>& getHistograms2D() const { return *mHist2D; }
  const std::vector<TH1D>& getHistograms1Dd() const { return *mHist1Dd; }
  const std::vector<TGraphAsymmErrors>& getGraphs() const { return *mHistGraph; }
  void resetHists();
  int32_t loadHistograms(std::vector<TH1F>& i1, std::vector<TH2F>& i2, std::vector<TH1D>& i3, std::vector<TGraphAsymmErrors>& i4, int32_t tasks = -1);
  void* AllocateScratchBuffer(size_t nBytes);

  static constexpr int32_t N_CLS_HIST = 8;
  static constexpr int32_t N_CLS_TYPE = 3;

  static constexpr int32_t MC_LABEL_INVALID = -1e9;

  enum QA_TASKS {
    taskTrackingEff = 1,
    taskTrackingRes = 2,
    taskTrackingResPull = 4,
    taskClusterAttach = 8,
    taskTrackStatistics = 16,
    taskClusterCounts = 32,
    taskDefault = 63,
    taskDefaultPostprocess = 31,
    tasksNoQC = 56
  };

 private:
  struct additionalMCParameters {
    float pt, phi, theta, eta, nWeightCls;
  };

  struct additionalClusterParameters {
    int32_t attached, fakeAttached, adjacent, fakeAdjacent;
    float pt;
  };

  int32_t InitQACreateHistograms();
  int32_t DoClusterCounts(uint64_t* attachClusterCounts, int32_t mode = 0);
  void PrintClusterCount(int32_t mode, int32_t& num, const char* name, uint64_t n, uint64_t normalization);
  void CopyO2MCtoIOPtr(GPUTrackingInOutPointers* ptr);
  template <class T>
  void SetAxisSize(T* e);
  void SetLegend(TLegend* l);
  double* CreateLogAxis(int32_t nbins, float xmin, float xmax);
  void ChangePadTitleSize(TPad* p, float size);
  void DrawHisto(TH1* histo, char* filename, char* options);
  void doPerfFigure(float x, float y, float size);
  void GetName(char* fname, int32_t k);
  template <class T>
  T* GetHist(T*& ee, std::vector<std::unique_ptr<TFile>>& tin, int32_t k, int32_t nNewInput);

  using mcInfo_t = GPUTPCMCInfo;
#ifdef GPUCA_TPC_GEOMETRY_O2
  mcLabels_t GetMCLabel(uint32_t i);
  mcLabel_t GetMCLabel(uint32_t i, uint32_t j);
#else
  struct mcLabelI_t {
    int32_t getTrackID() const { return AbsLabelID(track); }
    int32_t getEventID() const { return 0; }
    int32_t getSourceID() const { return 0; }
    int64_t getTrackEventSourceID() const { return getTrackID(); }
    bool isFake() const { return track < 0; }
    bool isValid() const { return track != MC_LABEL_INVALID; }
    void invalidate() { track = MC_LABEL_INVALID; }
    void setFakeFlag(bool v = true) { track = v ? FakeLabelID(track) : AbsLabelID(track); }
    void setNoise() { track = MC_LABEL_INVALID; }
    bool operator==(const mcLabel_t& l);
    bool operator!=(const mcLabel_t& l) { return !(*this == l); }
    mcLabelI_t() = default;
    mcLabelI_t(const mcLabel_t& l);
    int32_t track = MC_LABEL_INVALID;
  };
  const mcLabels_t& GetMCLabel(uint32_t i);
  const mcLabel_t& GetMCLabel(uint32_t i, uint32_t j);
  const mcInfo_t& GetMCTrack(const mcLabelI_t& label);
  static int32_t FakeLabelID(const int32_t id);
  static int32_t AbsLabelID(const int32_t id);
#endif
  template <class T>
  auto& GetMCTrackObj(T& obj, const mcLabelI_t& l);

  uint32_t GetNMCCollissions() const;
  uint32_t GetNMCTracks(int32_t iCol) const;
  uint32_t GetNMCTracks(const mcLabelI_t& label) const;
  uint32_t GetNMCLabels() const;
  const mcInfo_t& GetMCTrack(uint32_t iTrk, uint32_t iCol);
  const mcInfo_t& GetMCTrack(const mcLabel_t& label);
  int32_t GetMCLabelNID(const mcLabels_t& label);
  int32_t GetMCLabelNID(uint32_t i);
  int32_t GetMCLabelID(uint32_t i, uint32_t j);
  uint32_t GetMCLabelCol(uint32_t i, uint32_t j);
  static int32_t GetMCLabelID(const mcLabels_t& label, uint32_t j);
  static int32_t GetMCLabelID(const mcLabel_t& label);
  float GetMCLabelWeight(uint32_t i, uint32_t j);
  float GetMCLabelWeight(const mcLabels_t& label, uint32_t j);
  float GetMCLabelWeight(const mcLabel_t& label);
  const auto& GetClusterLabels();
  bool mcPresent();

  GPUChainTracking* mTracking;
  const GPUSettingsQA& mConfig;
  const GPUParam* mParam;

  const char* str_perf_figure_1 = "ALICE Performance 2018/03/20";
  // const char* str_perf_figure_2 = "2015, MC pp, #sqrt{s} = 5.02 TeV";
  const char* str_perf_figure_2 = "2015, MC Pb-Pb, #sqrt{s_{NN}} = 5.02 TeV";
  //-------------------------

  std::vector<mcLabelI_t> mTrackMCLabels;
#ifdef GPUCA_TPC_GEOMETRY_O2
  std::vector<std::vector<int32_t>> mTrackMCLabelsReverse;
  std::vector<std::vector<int32_t>> mRecTracks;
  std::vector<std::vector<int32_t>> mFakeTracks;
  std::vector<std::vector<additionalMCParameters>> mMCParam;
#else
  std::vector<int32_t> mTrackMCLabelsReverse[1];
  std::vector<int32_t> mRecTracks[1];
  std::vector<int32_t> mFakeTracks[1];
  std::vector<additionalMCParameters> mMCParam[1];
#endif
  std::vector<mcInfo_t> mMCInfos;
  std::vector<GPUTPCMCInfoCol> mMCInfosCol;
  std::vector<uint32_t> mMCNEvents;
  std::vector<uint32_t> mMCEventOffset;

  std::vector<additionalClusterParameters> mClusterParam;
  int32_t mNTotalFakes = 0;

  TH1F* mEff[4][2][2][5]; // eff,clone,fake,all - findable - secondaries - y,z,phi,eta,pt - work,result
  TGraphAsymmErrors* mEffResult[4][2][2][5];
  TCanvas* mCEff[6];
  TPad* mPEff[6][4];
  TLegend* mLEff[6];

  TH1F* mRes[5][5][2]; // y,z,phi,lambda,pt,ptlog res - param - res,mean
  TH2F* mRes2[5][5];
  TCanvas* mCRes[7];
  TPad* mPRes[7][5];
  TLegend* mLRes[6];

  TH1F* mPull[5][5][2]; // y,z,phi,lambda,pt,ptlog res - param - res,mean
  TH2F* mPull2[5][5];
  TCanvas* mCPull[7];
  TPad* mPPull[7][5];
  TLegend* mLPull[6];

  enum CL_types { CL_attached = 0,
                  CL_fake = 1,
                  CL_att_adj = 2,
                  CL_fakeAdj = 3,
                  CL_tracks = 4,
                  CL_physics = 5,
                  CL_prot = 6,
                  CL_all = 7 };
  TH1D* mClusters[N_CLS_TYPE * N_CLS_HIST - 1]; // attached, fakeAttached, attach+adjacent, fakeAdjacent, physics, protected, tracks, all / count, rel, integral
  TCanvas* mCClust[N_CLS_TYPE];
  TPad* mPClust[N_CLS_TYPE];
  TLegend* mLClust[N_CLS_TYPE];

  struct counts_t {
    int64_t nRejected = 0, nTube = 0, nTube200 = 0, nLoopers = 0, nLowPt = 0, n200MeV = 0, nPhysics = 0, nProt = 0, nUnattached = 0, nTotal = 0, nHighIncl = 0, nAbove400 = 0, nFakeRemove400 = 0, nFullFakeRemove400 = 0, nBelow40 = 0, nFakeProtect40 = 0, nMergedLooper = 0;
    double nUnaccessible = 0;
  } mClusterCounts;

  TH1F* mTracks;
  TCanvas* mCTracks;
  TPad* mPTracks;
  TLegend* mLTracks;

  TH1F* mNCl;
  TCanvas* mCNCl;
  TPad* mPNCl;
  TLegend* mLNCl;

  TH2F* mClXY;
  TCanvas* mCClXY;
  TPad* mPClXY;

  std::vector<TH2F*> mHistClusterCount;

  std::vector<TH1F>* mHist1D = nullptr;
  std::vector<TH2F>* mHist2D = nullptr;
  std::vector<TH1D>* mHist1Dd = nullptr;
  std::vector<TGraphAsymmErrors>* mHistGraph = nullptr;
  bool mHaveExternalHists = false;
  std::vector<TH1F**> mHist1D_pos{};
  std::vector<TH2F**> mHist2D_pos{};
  std::vector<TH1D**> mHist1Dd_pos{};
  std::vector<TGraphAsymmErrors**> mHistGraph_pos{};
  template <class T>
  auto getHistArray();
  template <class T, typename... Args>
  void createHist(T*& h, const char* name, Args... args);

  std::unique_ptr<GPUQAGarbageCollection> mGarbageCollector;
  template <class T, typename... Args>
  T* createGarbageCollected(Args... args);
  void clearGarbagageCollector();

  int32_t mNEvents = 0;
  bool mQAInitialized = false;
  bool mO2MCDataLoaded = false;
  int32_t mQATasks = 0;
  std::vector<std::vector<int32_t>> mcEffBuffer;
  std::vector<std::vector<int32_t>> mcLabelBuffer;
  std::vector<std::vector<bool>> mGoodTracks;
  std::vector<std::vector<bool>> mGoodHits;

  std::vector<uint64_t> mTrackingScratchBuffer;

  static std::vector<TColor*> mColors;
  static int32_t initColors();

  int32_t mMCTrackMin = -1, mMCTrackMax = -1;

  const o2::tpc::ClusterNativeAccess* mClNative = nullptr;
};

inline bool GPUQA::SuppressTrack(int32_t iTrack) const { return (mConfig.matchMCLabels.size() && !mGoodTracks[mNEvents][iTrack]); }
inline bool GPUQA::SuppressHit(int32_t iHit) const { return (mConfig.matchMCLabels.size() && !mGoodHits[mNEvents - 1][iHit]); }
inline int32_t GPUQA::HitAttachStatus(int32_t iHit) const { return (mClusterParam.size() && mClusterParam[iHit].fakeAttached ? (mClusterParam[iHit].attached ? 1 : 2) : 0); }

} // namespace GPUCA_NAMESPACE::gpu

#endif
#endif
