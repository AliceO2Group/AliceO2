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

#include "GPUQAConfig.h"
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
  GPUQA(GPUChainTracking* rec) {}
  ~GPUQA() = default;

  typedef structConfigQA configQA;

  int InitQA() { return 1; }
  void RunQA(bool matchOnly = false) {}
  int DrawQAHistograms() { return 1; }
  void SetMCTrackRange(int min, int max) {}
  bool SuppressTrack(int iTrack) const { return false; }
  bool SuppressHit(int iHit) const { return false; }
  bool HitAttachStatus(int iHit) const { return false; }
  int GetMCTrackLabel(unsigned int trackId) const { return -1; }
  bool clusterRemovable(int cid, bool prot) const { return false; }
  static bool QAAvailable() { return false; }
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else

#include "GPUTPCDef.h"
#include <cmath>

#ifdef GPUCA_TPC_GEOMETRY_O2
#include <gsl/span>
#endif

namespace o2
{
class MCCompLabel;
}

class AliHLTTPCClusterMCLabel;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUChainTracking;
class GPUTPCMCInfo;

class GPUQA
{
 public:
  GPUQA(GPUChainTracking* rec);
  ~GPUQA();

  typedef GPUQAConfig configQA;

  int InitQA();
  void RunQA(bool matchOnly = false);
  int DrawQAHistograms();
  void SetMCTrackRange(int min, int max);
  bool SuppressTrack(int iTrack) const;
  bool SuppressHit(int iHit) const;
  bool HitAttachStatus(int iHit) const;
  int GetMCTrackLabel(unsigned int trackId) const;
  bool clusterRemovable(int cid, bool prot) const;
  static bool QAAvailable() { return true; }

 private:
  struct additionalMCParameters {
    float pt, phi, theta, eta, nWeightCls;
  };

  struct additionalClusterParameters {
    int attached, fakeAttached, adjacent, fakeAdjacent;
    float pt;
  };

  void SetAxisSize(TH1F* e);
  void SetLegend(TLegend* l);
  double* CreateLogAxis(int nbins, float xmin, float xmax);
  void ChangePadTitleSize(TPad* p, float size);
  void DrawHisto(TH1* histo, char* filename, char* options);
  void doPerfFigure(float x, float y, float size);
  void GetName(char* fname, int k);
  template <class T>
  T* GetHist(T*& ee, std::vector<TFile*>& tin, int k, int nNewInput);

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
  bool mcPresent();

  static bool MCComp(const mcLabel_t& a, const mcLabel_t& b);

  GPUChainTracking* mTracking;
  const configQA& mConfig;

  //-------------------------: Some compile time settings....
  static const constexpr bool PLOT_ROOT = 0;
  static const constexpr bool FIX_SCALES = 0;
  static const constexpr bool PERF_FIGURE = 0;
  static const constexpr float FIXED_SCALES_MIN[5] = { -0.05, -0.05, -0.2, -0.2, -0.5 };
  static const constexpr float FIXED_SCALES_MAX[5] = { 0.4, 0.7, 5, 3, 6.5 };
  static const constexpr float LOG_PT_MIN = -1.;

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

  static constexpr int N_CLS_HIST = 8;
  static constexpr int N_CLS_TYPE = 3;
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

  long long int mNRecClustersRejected = 0, mNRecClustersTube = 0, mNRecClustersTube200 = 0, mNRecClustersLoopers = 0, mNRecClustersLowPt = 0, mNRecClusters200MeV = 0, mNRecClustersPhysics = 0, mNRecClustersProt = 0, mNRecClustersUnattached = 0, mNRecClustersTotal = 0, mNRecClustersHighIncl = 0,
                mNRecClustersAbove400 = 0, mNRecClustersFakeRemove400 = 0, mNRecClustersFullFakeRemove400 = 0, mNRecClustersBelow40 = 0, mNRecClustersFakeProtect40 = 0;
  double mNRecClustersUnaccessible = 0;

  TH1F* mTracks;
  TCanvas* mCTracks;
  TPad* mPTracks;
  TLegend* mLTracks;

  TH1F* mNCl;
  TCanvas* mCNCl;
  TPad* mPNCl;
  TLegend* mLNCl;

  int mNEvents = 0;
  bool mQAInitialized = false;
  std::vector<std::vector<int>> mcEffBuffer;
  std::vector<std::vector<int>> mcLabelBuffer;
  std::vector<std::vector<bool>> mGoodTracks;
  std::vector<std::vector<bool>> mGoodHits;

  static constexpr float Y_MAX = 40;
  static constexpr float Z_MAX = 100;
  static constexpr float PT_MIN = GPUCA_MIN_TRACK_PT_DEFAULT;
  static constexpr float PT_MIN2 = 0.1;
  static constexpr float PT_MIN_PRIM = 0.1;
  static constexpr float PT_MIN_CLUST = GPUCA_MIN_TRACK_PT_DEFAULT;
  static constexpr float PT_MAX = 20;
  static constexpr float ETA_MAX = 1.5;
  static constexpr float ETA_MAX2 = 0.9;

  static constexpr float MIN_WEIGHT_CLS = 40;
  static constexpr float FINDABLE_WEIGHT_CLS = 70;

  static constexpr int MC_LABEL_INVALID = -1e9;

  static constexpr bool CLUST_HIST_INT_SUM = false;

  static constexpr const int COLORCOUNT = 12;
  Color_t* mColorNums;

  static const constexpr char* EFF_TYPES[4] = { "Rec", "Clone", "Fake", "All" };
  static const constexpr char* FINDABLE_NAMES[2] = { "", "Findable" };
  static const constexpr char* PRIM_NAMES[2] = { "Prim", "Sec" };
  static const constexpr char* PARAMETER_NAMES[5] = { "Y", "Z", "#Phi", "#lambda", "Relative #it{p}_{T}" };
  static const constexpr char* PARAMETER_NAMES_NATIVE[5] = { "Y", "Z", "sin(#Phi)", "tan(#lambda)", "q/#it{p}_{T} (curvature)" };
  static const constexpr char* VSPARAMETER_NAMES[6] = { "Y", "Z", "Phi", "Eta", "Pt", "Pt_log" };
  static const constexpr char* EFF_NAMES[3] = { "Efficiency", "Clone Rate", "Fake Rate" };
  static const constexpr char* EFFICIENCY_TITLES[4] = { "Efficiency (Primary Tracks, Findable)", "Efficiency (Secondary Tracks, Findable)", "Efficiency (Primary Tracks)", "Efficiency (Secondary Tracks)" };
  static const constexpr double SCALE[5] = { 10., 10., 1000., 1000., 100. };
  static const constexpr double SCALE_NATIVE[5] = { 10., 10., 1000., 1000., 1. };
  static const constexpr char* XAXIS_TITLES[5] = { "#it{y}_{mc} (cm)", "#it{z}_{mc} (cm)", "#Phi_{mc} (rad)", "#eta_{mc}", "#it{p}_{Tmc} (GeV/#it{c})" };
  static const constexpr char* AXIS_TITLES[5] = { "#it{y}-#it{y}_{mc} (mm) (Resolution)", "#it{z}-#it{z}_{mc} (mm) (Resolution)", "#phi-#phi_{mc} (mrad) (Resolution)", "#lambda-#lambda_{mc} (mrad) (Resolution)", "(#it{p}_{T} - #it{p}_{Tmc}) / #it{p}_{Tmc} (%) (Resolution)" };
  static const constexpr char* AXIS_TITLES_NATIVE[5] = { "#it{y}-#it{y}_{mc} (mm) (Resolution)", "#it{z}-#it{z}_{mc} (mm) (Resolution)", "sin(#phi)-sin(#phi_{mc}) (Resolution)", "tan(#lambda)-tan(#lambda_{mc}) (Resolution)", "q*(q/#it{p}_{T} - q/#it{p}_{Tmc}) (Resolution)" };
  static const constexpr char* AXIS_TITLES_PULL[5] = { "#it{y}-#it{y}_{mc}/#sigma_{y} (Pull)", "#it{z}-#it{z}_{mc}/#sigma_{z} (Pull)", "sin(#phi)-sin(#phi_{mc})/#sigma_{sin(#phi)} (Pull)", "tan(#lambda)-tan(#lambda_{mc})/#sigma_{tan(#lambda)} (Pull)",
                                                       "q*(q/#it{p}_{T} - q/#it{p}_{Tmc})/#sigma_{q/#it{p}_{T}} (Pull)" };
  static const constexpr char* CLUSTER_NAMES[N_CLS_HIST] = { "Correctly attached clusters", "Fake attached clusters", "Attached + adjacent clusters", "Fake adjacent clusters", "Clusters of reconstructed tracks", "Used in Physics", "Protected", "All clusters" };
  static const constexpr char* CLUSTER_TITLES[N_CLS_TYPE] = { "Clusters Pt Distribution / Attachment", "Clusters Pt Distribution / Attachment (relative to all clusters)", "Clusters Pt Distribution / Attachment (integrated)" };
  static const constexpr char* CLUSTER_NAMES_SHORT[N_CLS_HIST] = { "Attached", "Fake", "AttachAdjacent", "FakeAdjacent", "FoundTracks", "Physics", "Protected", "All" };
  static const constexpr char* CLUSTER_TYPES[N_CLS_TYPE] = { "", "Ratio", "Integral" };
  static const constexpr int COLORS_HEX[COLORCOUNT] = { 0xB03030, 0x00A000, 0x0000C0, 0x9400D3, 0x19BBBF, 0xF25900, 0x7F7F7F, 0xFFD700, 0x07F707, 0x07F7F7, 0xF08080, 0x000000 };

  static const constexpr int CONFIG_DASHED_MARKERS = 0;

  static const constexpr float AXES_MIN[5] = { -Y_MAX, -Z_MAX, 0.f, -ETA_MAX, PT_MIN };
  static const constexpr float AXES_MAX[5] = { Y_MAX, Z_MAX, 2.f * M_PI, ETA_MAX, PT_MAX };
  static const constexpr int AXIS_BINS[5] = { 51, 51, 144, 31, 50 };
  static const constexpr int RES_AXIS_BINS[] = { 1017, 113 }; // Consecutive bin sizes, histograms are binned down until the maximum entry is 50, each bin size should evenly divide its predecessor.
  static const constexpr float RES_AXES[5] = { 1., 1., 0.03, 0.03, 1.0 };
  static const constexpr float RES_AXES_NATIVE[5] = { 1., 1., 0.1, 0.1, 5.0 };
  static const constexpr float PULL_AXIS = 10.f;

  int mMCTrackMin = -1, mMCTrackMax = -1;
};

inline bool GPUQA::SuppressTrack(int iTrack) const { return (mConfig.matchMCLabels.size() && !mGoodTracks[mNEvents][iTrack]); }
inline bool GPUQA::SuppressHit(int iHit) const { return (mConfig.matchMCLabels.size() && !mGoodHits[mNEvents - 1][iHit]); }
inline bool GPUQA::HitAttachStatus(int iHit) const { return (mClusterParam.size() && mClusterParam[iHit].fakeAttached ? (mClusterParam[iHit].attached ? 1 : 2) : 0); }

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif
