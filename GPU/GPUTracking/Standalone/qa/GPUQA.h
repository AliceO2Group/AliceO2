// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
typedef short int Color_t;

#if !defined(GPUCA_BUILD_QA) || defined(GPUCA_GPUCODE)

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUChainTracking;

class GPUQA
{
 public:
  GPUQA(GPUChainTracking* chain) {}
  ~GPUQA() = default;

  int InitQA(int tasks = 0) { return 1; }
  void RunQA(bool matchOnly = false) {}
  int DrawQAHistograms() { return 1; }
  void SetMCTrackRange(int min, int max) {}
  bool SuppressTrack(int iTrack) const { return false; }
  bool SuppressHit(int iHit) const { return false; }
  int HitAttachStatus(int iHit) const { return false; }
  int GetMCTrackLabel(unsigned int trackId) const { return -1; }
  bool clusterRemovable(int attach, bool prot) const { return false; }
  static bool QAAvailable() { return false; }
  static bool IsInitialized() { return false; }
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
class GPUParam;
struct GPUTPCMCInfo;
struct GPUQAGarbageCollection;

class GPUQA
{
 public:
  GPUQA();
  GPUQA(GPUChainTracking* chain, const GPUSettingsQA* config = nullptr, const GPUParam* param = nullptr);
  ~GPUQA();

  int InitQA(int tasks = -1);
  void RunQA(bool matchOnly = false, const std::vector<o2::tpc::TrackTPC>* tracksExternal = nullptr, const std::vector<o2::MCCompLabel>* tracksExtMC = nullptr, const o2::tpc::ClusterNativeAccess* clNative = nullptr);
  int DrawQAHistograms(TObjArray* qcout = nullptr);
  void DrawQAHistogramsCleanup(); // Needed after call to DrawQAHistograms with qcout != nullptr when GPUSettingsQA.shipToQCAsCanvas = true to clean up the Canvases etc.
  void SetMCTrackRange(int min, int max);
  bool SuppressTrack(int iTrack) const;
  bool SuppressHit(int iHit) const;
  int HitAttachStatus(int iHit) const;
  int GetMCTrackLabel(unsigned int trackId) const;
  bool clusterRemovable(int attach, bool prot) const;
  static bool QAAvailable() { return true; }
  bool IsInitialized() { return mQAInitialized; }

  const std::vector<TH1F>& getHistograms1D() const { return *mHist1D; }
  const std::vector<TH2F>& getHistograms2D() const { return *mHist2D; }
  const std::vector<TH1D>& getHistograms1Dd() const { return *mHist1Dd; }
  void resetHists();
  int loadHistograms(std::vector<TH1F>& i1, std::vector<TH2F>& i2, std::vector<TH1D>& i3, int tasks = -1);

  static constexpr int N_CLS_HIST = 8;
  static constexpr int N_CLS_TYPE = 3;

  static constexpr int MC_LABEL_INVALID = -1e9;

  enum QA_TASKS {
    taskTrackingEff = 1,
    taskTrackingRes = 2,
    taskTrackingResPull = 4,
    taskClusterAttach = 8,
    taskTrackStatistics = 16,
    taskClusterCounts = 32,
    taskDefault = 63,
    taskDefaultPostprocess = 31
  };

 private:
  struct additionalMCParameters {
    float pt, phi, theta, eta, nWeightCls;
  };

  struct additionalClusterParameters {
    int attached, fakeAttached, adjacent, fakeAdjacent;
    float pt;
  };

  int InitQACreateHistograms();
  int DoClusterCounts(unsigned long long int* attachClusterCounts, int mode = 0);
  void PrintClusterCount(int mode, int& num, const char* name, unsigned long long int n, unsigned long long int normalization);

  void SetAxisSize(TH1F* e);
  void SetLegend(TLegend* l);
  double* CreateLogAxis(int nbins, float xmin, float xmax);
  void ChangePadTitleSize(TPad* p, float size);
  void DrawHisto(TH1* histo, char* filename, char* options);
  void doPerfFigure(float x, float y, float size);
  void GetName(char* fname, int k);
  template <class T>
  T* GetHist(T*& ee, std::vector<std::unique_ptr<TFile>>& tin, int k, int nNewInput);

#ifdef GPUCA_TPC_GEOMETRY_O2
  using mcLabels_t = gsl::span<const o2::MCCompLabel>;
  using mcLabel_t = o2::MCCompLabel;
  using mcLabelI_t = mcLabel_t;
  using mcInfo_t = GPUTPCMCInfo;
  mcLabels_t GetMCLabel(unsigned int i);
  mcLabel_t GetMCLabel(unsigned int i, unsigned int j);
#else
  using mcLabels_t = AliHLTTPCClusterMCLabel;
  using mcLabel_t = AliHLTTPCClusterMCWeight;
  struct mcLabelI_t {
    int getTrackID() const { return AbsLabelID(track); }
    int getEventID() const { return 0; }
    long int getTrackEventSourceID() const { return getTrackID(); }
    bool isFake() const { return track < 0; }
    bool isValid() const { return track != MC_LABEL_INVALID; }
    void invalidate() { track = MC_LABEL_INVALID; }
    void setFakeFlag(bool v = true) { track = v ? FakeLabelID(track) : AbsLabelID(track); }
    void setNoise() { track = MC_LABEL_INVALID; }
    bool operator==(const mcLabel_t& l);
    bool operator!=(const mcLabel_t& l) { return !(*this == l); }
    mcLabelI_t() = default;
    mcLabelI_t(const mcLabel_t& l);
    int track = MC_LABEL_INVALID;
  };
  using mcInfo_t = GPUTPCMCInfo;
  const mcLabels_t& GetMCLabel(unsigned int i);
  const mcLabel_t& GetMCLabel(unsigned int i, unsigned int j);
  const mcInfo_t& GetMCTrack(const mcLabelI_t& label);
  static int FakeLabelID(const int id);
  static int AbsLabelID(const int id);
#endif
  template <class T>
  static auto& GetMCTrackObj(T& obj, const mcLabelI_t& l);

  unsigned int GetNMCCollissions();
  unsigned int GetNMCTracks(int iCol);
  unsigned int GetNMCLabels();
  const mcInfo_t& GetMCTrack(unsigned int iTrk, unsigned int iCol);
  const mcInfo_t& GetMCTrack(const mcLabel_t& label);
  int GetMCLabelNID(const mcLabels_t& label);
  int GetMCLabelNID(unsigned int i);
  int GetMCLabelID(unsigned int i, unsigned int j);
  int GetMCLabelCol(unsigned int i, unsigned int j);
  static int GetMCLabelID(const mcLabels_t& label, unsigned int j);
  static int GetMCLabelID(const mcLabel_t& label);
  float GetMCLabelWeight(unsigned int i, unsigned int j);
  float GetMCLabelWeight(const mcLabels_t& label, unsigned int j);
  float GetMCLabelWeight(const mcLabel_t& label);
  const auto& GetClusterLabels();
  bool mcPresent();

  static bool MCComp(const mcLabel_t& a, const mcLabel_t& b);

  GPUChainTracking* mTracking;
  const GPUSettingsQA& mConfig;
  const GPUParam& mParam;

  const char* str_perf_figure_1 = "ALICE Performance 2018/03/20";
  // const char* str_perf_figure_2 = "2015, MC pp, #sqrt{s} = 5.02 TeV";
  const char* str_perf_figure_2 = "2015, MC Pb-Pb, #sqrt{s_{NN}} = 5.02 TeV";
  //-------------------------

  std::vector<mcLabelI_t> mTrackMCLabels;
#ifdef GPUCA_TPC_GEOMETRY_O2
  std::vector<std::vector<int>> mTrackMCLabelsReverse;
  std::vector<std::vector<int>> mRecTracks;
  std::vector<std::vector<int>> mFakeTracks;
  std::vector<std::vector<additionalMCParameters>> mMCParam;
  std::vector<std::vector<mcInfo_t>> mMCInfos;
  std::vector<int> mNColTracks;
#else
  std::vector<int> mTrackMCLabelsReverse[1];
  std::vector<int> mRecTracks[1];
  std::vector<int> mFakeTracks[1];
  std::vector<additionalMCParameters> mMCParam[1];
#endif
  std::vector<additionalClusterParameters> mClusterParam;
  int mNTotalFakes = 0;

  TH1F* mEff[4][2][2][5][2]; // eff,clone,fake,all - findable - secondaries - y,z,phi,eta,pt - work,result
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
    long long int nRejected = 0, nTube = 0, nTube200 = 0, nLoopers = 0, nLowPt = 0, n200MeV = 0, nPhysics = 0, nProt = 0, nUnattached = 0, nTotal = 0, nHighIncl = 0, nAbove400 = 0, nFakeRemove400 = 0, nFullFakeRemove400 = 0, nBelow40 = 0, nFakeProtect40 = 0, nMergedLooper = 0;
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

  std::vector<TH2F*> mHistClusterCount;

  std::vector<TH1F>* mHist1D = nullptr;
  std::vector<TH2F>* mHist2D = nullptr;
  std::vector<TH1D>* mHist1Dd = nullptr;
  bool mHaveExternalHists = false;
  std::vector<TH1F**> mHist1D_pos{};
  std::vector<TH2F**> mHist2D_pos{};
  std::vector<TH1D**> mHist1Dd_pos{};
  template <class T>
  auto getHistArray();
  template <class T, typename... Args>
  void createHist(T*& h, const char* name, Args... args);

  std::unique_ptr<GPUQAGarbageCollection> mGarbageCollector;
  template <class T, typename... Args>
  T* createGarbageCollected(Args... args);
  void clearGarbagageCollector();

  int mNEvents = 0;
  bool mQAInitialized = false;
  int mQATasks = 0;
  std::vector<std::vector<int>> mcEffBuffer;
  std::vector<std::vector<int>> mcLabelBuffer;
  std::vector<std::vector<bool>> mGoodTracks;
  std::vector<std::vector<bool>> mGoodHits;

  static std::vector<TColor*> mColors;
  static int initColors();

  int mMCTrackMin = -1, mMCTrackMax = -1;

  const o2::tpc::ClusterNativeAccess* mClNative = nullptr;
};

inline bool GPUQA::SuppressTrack(int iTrack) const { return (mConfig.matchMCLabels.size() && !mGoodTracks[mNEvents][iTrack]); }
inline bool GPUQA::SuppressHit(int iHit) const { return (mConfig.matchMCLabels.size() && !mGoodHits[mNEvents - 1][iHit]); }
inline int GPUQA::HitAttachStatus(int iHit) const { return (mClusterParam.size() && mClusterParam[iHit].fakeAttached ? (mClusterParam[iHit].attached ? 1 : 2) : 0); }

} // namespace GPUCA_NAMESPACE::gpu

#endif
#endif
