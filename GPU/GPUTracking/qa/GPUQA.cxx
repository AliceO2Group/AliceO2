// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUQA.cxx
/// \author David Rohr

#define QA_DEBUG 0
#define QA_TIMING 0

#include "Rtypes.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TColor.h"
#include "TPaveText.h"
#include "TF1.h"
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TObjArray.h"
#include <sys/stat.h>

#include "GPUQA.h"
#include "GPUTPCDef.h"
#include "GPUTPCSliceData.h"
#include "GPUChainTracking.h"
#include "GPUTPCTrack.h"
#include "GPUTPCTracker.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPropagator.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTPCClusterData.h"
#include "GPUO2DataTypes.h"
#include "GPUParam.inc"
#include "GPUTPCClusterRejection.h"
#include "TPCFastTransform.h"
#ifdef HAVE_O2HEADERS
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#endif
#ifdef GPUCA_O2_LIB
#include "DetectorsRaw/HBFUtils.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/Constants.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Steer/MCKinematicsReader.h"
#include "TPDGCode.h"
#include "TParticlePDG.h"
#include "TDatabasePDG.h"
#endif
#include "GPUQAHelper.h"
#include <algorithm>
#include <cstdio>

#include "utils/qconfig.h"
#include "utils/timer.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifdef GPUCA_MERGER_BY_MC_LABEL
#define CHECK_CLUSTER_STATE_INIT_LEG_BY_MC()             \
  if (!unattached && mTrackMCLabels[id].isValid()) {     \
    int mcLabel = mTrackMCLabels[id].getTrackID();       \
    int mcEvent = mTrackMCLabels[id].getEventID();       \
    if (mTrackMCLabelsReverse[mcEvent][mcLabel] != id) { \
      attach &= (~gputpcgmmergertypes::attachGoodLeg);   \
    }                                                    \
  }
#else
#define CHECK_CLUSTER_STATE_INIT_LEG_BY_MC()
#endif

#define CHECK_CLUSTER_STATE_INIT()                                        \
  bool unattached = attach == 0;                                          \
  float qpt = 0;                                                          \
  bool lowPt = false;                                                     \
  bool mev200 = false;                                                    \
  bool mergedLooper = false;                                              \
  int id = attach & gputpcgmmergertypes::attachTrackMask;                 \
  if (!unattached) {                                                      \
    qpt = fabsf(mTracking->mIOPtrs.mergedTracks[id].GetParam().GetQPt()); \
    lowPt = qpt > mTracking->GetParam().rec.tpcRejectQPt;                 \
    mev200 = qpt > 5;                                                     \
    mergedLooper = mTracking->mIOPtrs.mergedTracks[id].MergedLooper();    \
  }                                                                       \
  bool physics = false, protect = false;                                  \
  CHECK_CLUSTER_STATE_INIT_LEG_BY_MC();

#define CHECK_CLUSTER_STATE()                                                                              \
  CHECK_CLUSTER_STATE_INIT()                                                                               \
  if (mev200) {                                                                                            \
    mClusterCounts.n200MeV++;                                                                              \
  }                                                                                                        \
  if (lowPt) {                                                                                             \
    mClusterCounts.nLowPt++;                                                                               \
  } else if (mergedLooper) {                                                                               \
    mClusterCounts.nMergedLooper++;                                                                        \
  } else {                                                                                                 \
    GPUTPCClusterRejection::GetProtectionStatus<true>(attach, physics, protect, &mClusterCounts, &mev200); \
  }

#define CHECK_CLUSTER_STATE_NOCOUNT()                                             \
  CHECK_CLUSTER_STATE_INIT()                                                      \
  (void)mev200; /* silence unused variable warning*/                              \
  if (!lowPt && !mergedLooper) {                                                  \
    GPUTPCClusterRejection::GetProtectionStatus<false>(attach, physics, protect); \
  }

static const GPUSettingsQA& GPUQA_GetConfig(GPUChainTracking* chain)
{
  static GPUSettingsQA defaultConfig;
  if (chain && chain->mConfigQA) {
    return *chain->mConfigQA;
  } else {
    return defaultConfig;
  }
}

static const constexpr bool PLOT_ROOT = 0;
static const constexpr bool FIX_SCALES = 0;
static const constexpr bool PERF_FIGURE = 0;
static const constexpr float FIXED_SCALES_MIN[5] = {-0.05, -0.05, -0.2, -0.2, -0.5};
static const constexpr float FIXED_SCALES_MAX[5] = {0.4, 0.7, 5, 3, 6.5};
static const constexpr float LOG_PT_MIN = -1.;

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

static constexpr bool CLUST_HIST_INT_SUM = false;

static constexpr const int COLORCOUNT = 12;

static const constexpr char* EFF_TYPES[4] = {"Rec", "Clone", "Fake", "All"};
static const constexpr char* FINDABLE_NAMES[2] = {"", "Findable"};
static const constexpr char* PRIM_NAMES[2] = {"Prim", "Sec"};
static const constexpr char* PARAMETER_NAMES[5] = {"Y", "Z", "#Phi", "#lambda", "Relative #it{p}_{T}"};
static const constexpr char* PARAMETER_NAMES_NATIVE[5] = {"Y", "Z", "sin(#Phi)", "tan(#lambda)", "q/#it{p}_{T} (curvature)"};
static const constexpr char* VSPARAMETER_NAMES[6] = {"Y", "Z", "Phi", "Eta", "Pt", "Pt_log"};
static const constexpr char* EFF_NAMES[3] = {"Efficiency", "Clone Rate", "Fake Rate"};
static const constexpr char* EFFICIENCY_TITLES[4] = {"Efficiency (Primary Tracks, Findable)", "Efficiency (Secondary Tracks, Findable)", "Efficiency (Primary Tracks)", "Efficiency (Secondary Tracks)"};
static const constexpr double SCALE[5] = {10., 10., 1000., 1000., 100.};
static const constexpr double SCALE_NATIVE[5] = {10., 10., 1000., 1000., 1.};
static const constexpr char* XAXIS_TITLES[5] = {"#it{y}_{mc} (cm)", "#it{z}_{mc} (cm)", "#Phi_{mc} (rad)", "#eta_{mc}", "#it{p}_{Tmc} (GeV/#it{c})"};
static const constexpr char* AXIS_TITLES[5] = {"#it{y}-#it{y}_{mc} (mm) (Resolution)", "#it{z}-#it{z}_{mc} (mm) (Resolution)", "#phi-#phi_{mc} (mrad) (Resolution)", "#lambda-#lambda_{mc} (mrad) (Resolution)", "(#it{p}_{T} - #it{p}_{Tmc}) / #it{p}_{Tmc} (%) (Resolution)"};
static const constexpr char* AXIS_TITLES_NATIVE[5] = {"#it{y}-#it{y}_{mc} (mm) (Resolution)", "#it{z}-#it{z}_{mc} (mm) (Resolution)", "sin(#phi)-sin(#phi_{mc}) (Resolution)", "tan(#lambda)-tan(#lambda_{mc}) (Resolution)", "q*(q/#it{p}_{T} - q/#it{p}_{Tmc}) (Resolution)"};
static const constexpr char* AXIS_TITLES_PULL[5] = {"#it{y}-#it{y}_{mc}/#sigma_{y} (Pull)", "#it{z}-#it{z}_{mc}/#sigma_{z} (Pull)", "sin(#phi)-sin(#phi_{mc})/#sigma_{sin(#phi)} (Pull)", "tan(#lambda)-tan(#lambda_{mc})/#sigma_{tan(#lambda)} (Pull)",
                                                    "q*(q/#it{p}_{T} - q/#it{p}_{Tmc})/#sigma_{q/#it{p}_{T}} (Pull)"};
static const constexpr char* CLUSTER_NAMES[GPUQA::N_CLS_HIST] = {"Correctly attached clusters", "Fake attached clusters", "Attached + adjacent clusters", "Fake adjacent clusters", "Clusters of reconstructed tracks", "Used in Physics", "Protected", "All clusters"};
static const constexpr char* CLUSTER_TITLES[GPUQA::N_CLS_TYPE] = {"Clusters Pt Distribution / Attachment", "Clusters Pt Distribution / Attachment (relative to all clusters)", "Clusters Pt Distribution / Attachment (integrated)"};
static const constexpr char* CLUSTER_NAMES_SHORT[GPUQA::N_CLS_HIST] = {"Attached", "Fake", "AttachAdjacent", "FakeAdjacent", "FoundTracks", "Physics", "Protected", "All"};
static const constexpr char* CLUSTER_TYPES[GPUQA::N_CLS_TYPE] = {"", "Ratio", "Integral"};
static const constexpr int COLORS_HEX[COLORCOUNT] = {0xB03030, 0x00A000, 0x0000C0, 0x9400D3, 0x19BBBF, 0xF25900, 0x7F7F7F, 0xFFD700, 0x07F707, 0x07F7F7, 0xF08080, 0x000000};

static const constexpr int CONFIG_DASHED_MARKERS = 0;

static const constexpr float AXES_MIN[5] = {-Y_MAX, -Z_MAX, 0.f, -ETA_MAX, PT_MIN};
static const constexpr float AXES_MAX[5] = {Y_MAX, Z_MAX, 2.f * M_PI, ETA_MAX, PT_MAX};
static const constexpr int AXIS_BINS[5] = {51, 51, 144, 31, 50};
static const constexpr int RES_AXIS_BINS[] = {1017, 113}; // Consecutive bin sizes, histograms are binned down until the maximum entry is 50, each bin size should evenly divide its predecessor.
static const constexpr float RES_AXES[5] = {1., 1., 0.03, 0.03, 1.0};
static const constexpr float RES_AXES_NATIVE[5] = {1., 1., 0.1, 0.1, 5.0};
static const constexpr float PULL_AXIS = 10.f;

std::vector<TColor*> GPUQA::mColors;
int GPUQA::initColors()
{
  mColors.reserve(COLORCOUNT);
  for (int i = 0; i < COLORCOUNT; i++) {
    float f1 = (float)((COLORS_HEX[i] >> 16) & 0xFF) / (float)0xFF;
    float f2 = (float)((COLORS_HEX[i] >> 8) & 0xFF) / (float)0xFF;
    float f3 = (float)((COLORS_HEX[i] >> 0) & 0xFF) / (float)0xFF;
    mColors.emplace_back(new TColor(10000 + i, f1, f2, f3));
  }
  return 0;
}
static constexpr Color_t defaultColorNUms[COLORCOUNT] = {kRed, kBlue, kGreen, kMagenta, kOrange, kAzure, kBlack, kYellow, kGray, kTeal, kSpring, kPink};

#define TRACK_EXPECTED_REFERENCE_X_DEFAULT 81
#ifdef GPUCA_TPC_GEOMETRY_O2
inline unsigned int GPUQA::GetNMCCollissions()
{
  return mNColTracks.size();
}
inline unsigned int GPUQA::GetNMCTracks(int iCol) { return mNColTracks[iCol]; }
inline unsigned int GPUQA::GetNMCLabels() { return mClNative->clustersMCTruth ? mClNative->clustersMCTruth->getIndexedSize() : 0; }
inline const GPUQA::mcInfo_t& GPUQA::GetMCTrack(unsigned int iTrk, unsigned int iCol) { return mMCInfos[iCol][iTrk]; }
inline const GPUQA::mcInfo_t& GPUQA::GetMCTrack(const mcLabel_t& label) { return mMCInfos[label.getEventID()][label.getTrackID()]; }
inline GPUQA::mcLabels_t GPUQA::GetMCLabel(unsigned int i) { return mClNative->clustersMCTruth->getLabels(i); }
inline int GPUQA::GetMCLabelNID(const mcLabels_t& label) { return label.size(); }
inline int GPUQA::GetMCLabelNID(unsigned int i) { return mClNative->clustersMCTruth->getLabels(i).size(); }
inline GPUQA::mcLabel_t GPUQA::GetMCLabel(unsigned int i, unsigned int j) { return mClNative->clustersMCTruth->getLabels(i)[j]; }
inline int GPUQA::GetMCLabelID(unsigned int i, unsigned int j) { return mClNative->clustersMCTruth->getLabels(i)[j].getTrackID(); }
inline int GPUQA::GetMCLabelID(const mcLabels_t& label, unsigned int j) { return label[j].getTrackID(); }
inline int GPUQA::GetMCLabelID(const mcLabel_t& label) { return label.getTrackID(); }
inline int GPUQA::GetMCLabelCol(unsigned int i, unsigned int j) { return mClNative->clustersMCTruth->getLabels(i)[j].getEventID(); }
inline const auto& GPUQA::GetClusterLabels() { return mClNative->clustersMCTruth; }
inline float GPUQA::GetMCLabelWeight(unsigned int i, unsigned int j) { return 1; }
inline float GPUQA::GetMCLabelWeight(const mcLabels_t& label, unsigned int j) { return 1; }
inline float GPUQA::GetMCLabelWeight(const mcLabel_t& label) { return 1; }
inline bool GPUQA::mcPresent() { return !mConfig.noMC && mTracking && mClNative->clustersMCTruth && mNColTracks.size(); }
#define TRACK_EXPECTED_REFERENCE_X 78
#else
inline GPUQA::mcLabelI_t::mcLabelI_t(const GPUQA::mcLabel_t& l) : track(l.fMCID)
{
}
inline bool GPUQA::mcLabelI_t::operator==(const GPUQA::mcLabel_t& l) { return AbsLabelID(track) == l.fMCID; }
inline unsigned int GPUQA::GetNMCCollissions() { return 1; }
inline unsigned int GPUQA::GetNMCTracks(int iCol) { return mTracking->mIOPtrs.nMCInfosTPC; }
inline unsigned int GPUQA::GetNMCLabels() { return mTracking->mIOPtrs.nMCLabelsTPC; }
inline const GPUQA::mcInfo_t& GPUQA::GetMCTrack(unsigned int iTrk, unsigned int iCol) { return mTracking->mIOPtrs.mcInfosTPC[AbsLabelID(iTrk)]; }
inline const GPUQA::mcInfo_t& GPUQA::GetMCTrack(const mcLabel_t& label) { return GetMCTrack(label.fMCID, 0); }
inline const GPUQA::mcInfo_t& GPUQA::GetMCTrack(const mcLabelI_t& label) { return GetMCTrack(label.track, 0); }
inline const GPUQA::mcLabels_t& GPUQA::GetMCLabel(unsigned int i) { return mTracking->mIOPtrs.mcLabelsTPC[i]; }
inline const GPUQA::mcLabel_t& GPUQA::GetMCLabel(unsigned int i, unsigned int j) { return mTracking->mIOPtrs.mcLabelsTPC[i].fClusterID[j]; }
inline int GPUQA::GetMCLabelNID(const mcLabels_t& label) { return 3; }
inline int GPUQA::GetMCLabelNID(unsigned int i) { return 3; }
inline int GPUQA::GetMCLabelID(unsigned int i, unsigned int j) { return mTracking->mIOPtrs.mcLabelsTPC[i].fClusterID[j].fMCID; }
inline int GPUQA::GetMCLabelID(const mcLabels_t& label, unsigned int j) { return label.fClusterID[j].fMCID; }
inline int GPUQA::GetMCLabelID(const mcLabel_t& label) { return label.fMCID; }
inline int GPUQA::GetMCLabelCol(unsigned int i, unsigned int j) { return 0; }
inline const auto& GPUQA::GetClusterLabels() { return mTracking->mIOPtrs.mcLabelsTPC; }
inline float GPUQA::GetMCLabelWeight(unsigned int i, unsigned int j) { return mTracking->mIOPtrs.mcLabelsTPC[i].fClusterID[j].fWeight; }
inline float GPUQA::GetMCLabelWeight(const mcLabels_t& label, unsigned int j) { return label.fClusterID[j].fWeight; }
inline float GPUQA::GetMCLabelWeight(const mcLabel_t& label) { return label.fWeight; }
inline int GPUQA::FakeLabelID(int id) { return id < 0 ? id : (-2 - id); }
inline int GPUQA::AbsLabelID(int id) { return id >= 0 ? id : (-id - 2); }
inline bool GPUQA::mcPresent() { return !mConfig.noMC && mTracking && GetNMCLabels() && GetNMCTracks(0); }
#define TRACK_EXPECTED_REFERENCE_X TRACK_EXPECTED_REFERENCE_X_DEFAULT
#endif
template <class T>
inline auto& GPUQA::GetMCTrackObj(T& obj, const GPUQA::mcLabelI_t& l)
{
  return obj[l.getEventID()][l.getTrackID()];
}

template <>
auto GPUQA::getHistArray<TH1F>()
{
  return std::make_pair(mHist1D, &mHist1D_pos);
}
template <>
auto GPUQA::getHistArray<TH2F>()
{
  return std::make_pair(mHist2D, &mHist2D_pos);
}
template <>
auto GPUQA::getHistArray<TH1D>()
{
  return std::make_pair(mHist1Dd, &mHist1Dd_pos);
}
template <class T, typename... Args>
void GPUQA::createHist(T*& h, const char* name, Args... args)
{
  const auto& p = getHistArray<T>();
  if (mHaveExternalHists) {
    if (p.first->size() <= p.second->size()) {
      throw std::runtime_error("Incoming histogram array incomplete");
    }
    if (strcmp((*p.first)[p.second->size()].GetName(), name)) {
      throw std::runtime_error("Incoming histogram has incorrect name");
    }
  } else {
    p.first->emplace_back(name, args...);
  }
  p.second->emplace_back(&h);
  h = &p.first->back();
}

namespace GPUCA_NAMESPACE::gpu
{
struct GPUQAGarbageCollection {
  std::tuple<std::vector<std::unique_ptr<TCanvas>>, std::vector<std::unique_ptr<TLegend>>, std::vector<std::unique_ptr<TPad>>, std::vector<std::unique_ptr<TLatex>>, std::vector<std::unique_ptr<TH1D>>> v;
};
} // namespace GPUCA_NAMESPACE::gpu

template <class T, typename... Args>
T* GPUQA::createGarbageCollected(Args... args)
{
  auto& v = std::get<std::vector<std::unique_ptr<T>>>(mGarbageCollector->v);
  v.emplace_back(std::make_unique<T>(args...));
  return v.back().get();
}
void GPUQA::clearGarbagageCollector()
{
  std::get<std::vector<std::unique_ptr<TPad>>>(mGarbageCollector->v).clear(); // Make sure to depete TPad first due to ROOT ownership (std::tuple has no defined order in its destructor)
  std::apply([](auto&&... args) { ((args.clear()), ...); }, mGarbageCollector->v);
}

GPUQA::GPUQA(GPUChainTracking* chain, const GPUSettingsQA* config, const GPUParam* param) : mTracking(chain), mConfig(config ? *config : GPUQA_GetConfig(chain)), mParam(param ? *param : chain->GetParam()), mGarbageCollector(std::make_unique<GPUQAGarbageCollection>())
{
  static int initColorsInitialized = initColors();
  (void)initColorsInitialized;
}

GPUQA::~GPUQA()
{
  if (mQAInitialized && !mHaveExternalHists) {
    delete mHist1D;
    delete mHist2D;
    delete mHist1Dd;
  }
  clearGarbagageCollector(); // Needed to guarantee correct order for ROOT ownership
}

inline bool GPUQA::MCComp(const mcLabel_t& a, const mcLabel_t& b) { return (GPUQA::GetMCLabelID(a) > GPUQA::GetMCLabelID(b)); }

bool GPUQA::clusterRemovable(int attach, bool prot) const
{
  CHECK_CLUSTER_STATE_NOCOUNT();
  if (prot) {
    return protect || physics;
  }
  return (!unattached && !physics && !protect);
}

void GPUQA::SetAxisSize(TH1F* e)
{
  e->GetYaxis()->SetTitleOffset(1.0);
  e->GetYaxis()->SetTitleSize(0.045);
  e->GetYaxis()->SetLabelSize(0.045);
  e->GetXaxis()->SetTitleOffset(1.03);
  e->GetXaxis()->SetTitleSize(0.045);
  e->GetXaxis()->SetLabelOffset(-0.005);
  e->GetXaxis()->SetLabelSize(0.045);
}

void GPUQA::SetLegend(TLegend* l)
{
  l->SetTextFont(72);
  l->SetTextSize(0.016);
  l->SetFillColor(0);
}

double* GPUQA::CreateLogAxis(int nbins, float xmin, float xmax)
{
  float logxmin = std::log10(xmin);
  float logxmax = std::log10(xmax);
  float binwidth = (logxmax - logxmin) / nbins;

  double* xbins = new double[nbins + 1];

  xbins[0] = xmin;
  for (int i = 1; i <= nbins; i++) {
    xbins[i] = std::pow(10, logxmin + i * binwidth);
  }
  return xbins;
}

void GPUQA::ChangePadTitleSize(TPad* p, float size)
{
  p->Update();
  TPaveText* pt = (TPaveText*)(p->GetPrimitive("title"));
  if (pt == nullptr) {
    GPUError("Error changing title");
  } else {
    pt->SetTextSize(size);
    p->Modified();
  }
}

void GPUQA::DrawHisto(TH1* histo, char* filename, char* options)
{
  TCanvas tmp;
  tmp.cd();
  histo->Draw(options);
  tmp.Print(filename);
}

void GPUQA::doPerfFigure(float x, float y, float size)
{
  if (!PERF_FIGURE) {
    return;
  }
  TLatex* t = createGarbageCollected<TLatex>();
  t->SetNDC(kTRUE);
  t->SetTextColor(1);
  t->SetTextSize(size);
  t->DrawLatex(x, y, str_perf_figure_1);
  t->DrawLatex(x, y - 0.01 - size, str_perf_figure_2);
}

void GPUQA::SetMCTrackRange(int min, int max)
{
  mMCTrackMin = min;
  mMCTrackMax = max;
}

int GPUQA::GetMCTrackLabel(unsigned int trackId) const { return (trackId >= mTrackMCLabels.size() ? MC_LABEL_INVALID : mTrackMCLabels[trackId].getTrackID()); }

int GPUQA::InitQACreateHistograms()
{
  char name[2048], fname[1024];
  if (mQATasks & taskTrackingEff) {
    // Create Efficiency Histograms
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 5; l++) {
            for (int m = 0; m < 2; m++) {
              sprintf(name, "%s%s%s%sVs%s", m ? "eff" : "tracks", EFF_TYPES[i], FINDABLE_NAMES[j], PRIM_NAMES[k], VSPARAMETER_NAMES[l]);
              if (l == 4) {
                std::unique_ptr<double[]> binsPt{CreateLogAxis(AXIS_BINS[4], k == 0 ? PT_MIN_PRIM : AXES_MIN[4], AXES_MAX[4])};
                createHist(mEff[i][j][k][l][m], name, name, AXIS_BINS[l], binsPt.get());
              } else {
                createHist(mEff[i][j][k][l][m], name, name, AXIS_BINS[l], AXES_MIN[l], AXES_MAX[l]);
              }
              if (!mHaveExternalHists) {
                mEff[i][j][k][l][m]->Sumw2();
              }
            }
          }
        }
      }
    }
  }

  // Create Resolution Histograms
  if (mQATasks & taskTrackingRes) {
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
        sprintf(name, "rms_%s_vs_%s", VSPARAMETER_NAMES[i], VSPARAMETER_NAMES[j]);
        sprintf(fname, "mean_%s_vs_%s", VSPARAMETER_NAMES[i], VSPARAMETER_NAMES[j]);
        if (j == 4) {
          std::unique_ptr<double[]> binsPt{CreateLogAxis(AXIS_BINS[4], mConfig.resPrimaries == 1 ? PT_MIN_PRIM : AXES_MIN[4], AXES_MAX[4])};
          createHist(mRes[i][j][0], name, name, AXIS_BINS[j], binsPt.get());
          createHist(mRes[i][j][1], fname, fname, AXIS_BINS[j], binsPt.get());
        } else {
          createHist(mRes[i][j][0], name, name, AXIS_BINS[j], AXES_MIN[j], AXES_MAX[j]);
          createHist(mRes[i][j][1], fname, fname, AXIS_BINS[j], AXES_MIN[j], AXES_MAX[j]);
        }
        sprintf(name, "res_%s_vs_%s", VSPARAMETER_NAMES[i], VSPARAMETER_NAMES[j]);
        const float* axis = mConfig.nativeFitResolutions ? RES_AXES_NATIVE : RES_AXES;
        const int nbins = i == 4 && mConfig.nativeFitResolutions ? (10 * RES_AXIS_BINS[0]) : RES_AXIS_BINS[0];
        if (j == 4) {
          std::unique_ptr<double[]> binsPt{CreateLogAxis(AXIS_BINS[4], mConfig.resPrimaries == 1 ? PT_MIN_PRIM : AXES_MIN[4], AXES_MAX[4])};
          createHist(mRes2[i][j], name, name, nbins, -axis[i], axis[i], AXIS_BINS[j], binsPt.get());
        } else {
          createHist(mRes2[i][j], name, name, nbins, -axis[i], axis[i], AXIS_BINS[j], AXES_MIN[j], AXES_MAX[j]);
        }
      }
    }
  }

  // Create Pull Histograms
  if (mQATasks & taskTrackingResPull) {
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
        sprintf(name, "pull_rms_%s_vs_%s", VSPARAMETER_NAMES[i], VSPARAMETER_NAMES[j]);
        sprintf(fname, "pull_mean_%s_vs_%s", VSPARAMETER_NAMES[i], VSPARAMETER_NAMES[j]);
        if (j == 4) {
          std::unique_ptr<double[]> binsPt{CreateLogAxis(AXIS_BINS[4], AXES_MIN[4], AXES_MAX[4])};
          createHist(mPull[i][j][0], name, name, AXIS_BINS[j], binsPt.get());
          createHist(mPull[i][j][1], fname, fname, AXIS_BINS[j], binsPt.get());
        } else {
          createHist(mPull[i][j][0], name, name, AXIS_BINS[j], AXES_MIN[j], AXES_MAX[j]);
          createHist(mPull[i][j][1], fname, fname, AXIS_BINS[j], AXES_MIN[j], AXES_MAX[j]);
        }
        sprintf(name, "pull_%s_vs_%s", VSPARAMETER_NAMES[i], VSPARAMETER_NAMES[j]);
        if (j == 4) {
          std::unique_ptr<double[]> binsPt{CreateLogAxis(AXIS_BINS[4], AXES_MIN[4], AXES_MAX[4])};
          createHist(mPull2[i][j], name, name, RES_AXIS_BINS[0], -PULL_AXIS, PULL_AXIS, AXIS_BINS[j], binsPt.get());
        } else {
          createHist(mPull2[i][j], name, name, RES_AXIS_BINS[0], -PULL_AXIS, PULL_AXIS, AXIS_BINS[j], AXES_MIN[j], AXES_MAX[j]);
        }
      }
    }
  }

  // Create Cluster Histograms
  if (mQATasks & taskClusterAttach) {
    for (int i = 0; i < N_CLS_TYPE * N_CLS_HIST - 1; i++) {
      int ioffset = i >= (2 * N_CLS_HIST - 1) ? (2 * N_CLS_HIST - 1) : i >= N_CLS_HIST ? N_CLS_HIST : 0;
      int itype = i >= (2 * N_CLS_HIST - 1) ? 2 : i >= N_CLS_HIST ? 1 : 0;
      sprintf(name, "clusters%s%s", CLUSTER_NAMES_SHORT[i - ioffset], CLUSTER_TYPES[itype]);
      std::unique_ptr<double[]> binsPt{CreateLogAxis(AXIS_BINS[4], PT_MIN_CLUST, PT_MAX)};
      createHist(mClusters[i], name, name, AXIS_BINS[4], binsPt.get());
    }
  }

  if (mQATasks & taskTrackStatistics) {
    // Create Tracks Histograms
    sprintf(name, "nclusters");
    createHist(mNCl, name, name, 160, 0, 159);
    sprintf(name, "tracks");
    std::unique_ptr<double[]> binsPt{CreateLogAxis(AXIS_BINS[4], PT_MIN_CLUST, PT_MAX)};
    createHist(mTracks, name, name, AXIS_BINS[4], binsPt.get());
  }

  if ((mQATasks & taskClusterCounts) && mConfig.clusterRejectionHistograms) {
    int num = DoClusterCounts(nullptr, 2);
    mHistClusterCount.resize(num);
    DoClusterCounts(nullptr, 1);
  }

  for (unsigned int i = 0; i < mHist1D->size(); i++) {
    *mHist1D_pos[i] = &(*mHist1D)[i];
  }
  for (unsigned int i = 0; i < mHist2D->size(); i++) {
    *mHist2D_pos[i] = &(*mHist2D)[i];
  }
  for (unsigned int i = 0; i < mHist1Dd->size(); i++) {
    *mHist1Dd_pos[i] = &(*mHist1Dd)[i];
  }

  return 0;
}

int GPUQA::loadHistograms(std::vector<TH1F>& i1, std::vector<TH2F>& i2, std::vector<TH1D>& i3, int tasks)
{
  if (tasks == -1) {
    tasks = taskDefaultPostprocess;
  }
  if (mQAInitialized && (!mHaveExternalHists || tasks != mQATasks)) {
    throw std::runtime_error("QA not initialized or initialized with different task array");
  }
  mHist1D = &i1;
  mHist2D = &i2;
  mHist1Dd = &i3;
  mHist1D_pos.clear();
  mHist2D_pos.clear();
  mHist1Dd_pos.clear();
  mHaveExternalHists = true;
  mQATasks = tasks;
  if (InitQACreateHistograms()) {
    return 1;
  }
  mQAInitialized = true;
  return 0;
}

int GPUQA::InitQA(int tasks)
{
  if (mQAInitialized) {
    throw std::runtime_error("QA already initialized");
  }
  if (tasks == -1) {
    tasks = taskDefault;
  }

  mHist1D = new std::vector<TH1F>;
  mHist2D = new std::vector<TH2F>;
  mHist1Dd = new std::vector<TH1D>;
  mQATasks = tasks;

  if (mTracking) {
    mClNative = mTracking->mIOPtrs.clustersNative;
  }

  if (InitQACreateHistograms()) {
    return 1;
  }

  if (mConfig.enableLocalOutput) {
    mkdir("plots", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

#ifdef GPUCA_O2_LIB
  if (!mConfig.noMC) {
    HighResTimer timer;
    if (mTracking && mTracking->GetProcessingSettings().debugLevel) {
      GPUInfo("Start reading O2 Track MC information");
      timer.Start();
    }
    static constexpr float PRIM_MAX_T = 0.01f;

    o2::steer::MCKinematicsReader mcReader("collisioncontext.root");
    int nSimEvents = mcReader.getNEvents(0);
    mTrackMCLabelsReverse.resize(nSimEvents);
    mRecTracks.resize(nSimEvents);
    mFakeTracks.resize(nSimEvents);
    mMCParam.resize(nSimEvents);
    mMCInfos.resize(nSimEvents);
    mNColTracks.resize(nSimEvents);
    std::vector<int> refId;

    auto dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
    auto evrec = dc->getEventRecords();

    for (int i = 0; i < nSimEvents; i++) {
      auto ir = evrec[i];
      auto ir0 = o2::raw::HBFUtils::Instance().getFirstIRofTF(ir);
      float timebin = (float)ir.differenceInBC(ir0) / o2::tpc::constants::LHCBCPERTIMEBIN;

      const std::vector<o2::MCTrack>& tracks = mcReader.getTracks(0, i);
      const std::vector<o2::TrackReference>& trackRefs = mcReader.getTrackRefsByEvent(0, i);

      refId.resize(tracks.size());
      std::fill(refId.begin(), refId.end(), -1);
      for (unsigned int j = 0; j < trackRefs.size(); j++) {
        if (trackRefs[j].getDetectorId() == o2::detectors::DetID::TPC) {
          int trkId = trackRefs[j].getTrackID();
          if (refId[trkId] == -1) {
            refId[trkId] = j;
          }
        }
      }
      mNColTracks[i] = tracks.size();
      mMCInfos[i].resize(tracks.size());
      for (unsigned int j = 0; j < tracks.size(); j++) {
        auto& info = mMCInfos[i][j];
        const auto& trk = tracks[j];
        TParticlePDG* particle = TDatabasePDG::Instance()->GetParticle(trk.GetPdgCode());
        Int_t pid = -1;
        if (abs(trk.GetPdgCode()) == kElectron) {
          pid = 0;
        }
        if (abs(trk.GetPdgCode()) == kMuonMinus) {
          pid = 1;
        }
        if (abs(trk.GetPdgCode()) == kPiPlus) {
          pid = 2;
        }
        if (abs(trk.GetPdgCode()) == kKPlus) {
          pid = 3;
        }
        if (abs(trk.GetPdgCode()) == kProton) {
          pid = 4;
        }

        info.charge = particle ? particle->Charge() : 0;
        info.prim = trk.T() < PRIM_MAX_T;
        info.primDaughters = 0;
        if (trk.getFirstDaughterTrackId() != -1) {
          for (int k = trk.getFirstDaughterTrackId(); k <= trk.getLastDaughterTrackId(); k++) {
            if (tracks[k].T() < PRIM_MAX_T) {
              info.primDaughters = 1;
              break;
            }
          }
        }
        info.pid = pid;
        info.t0 = timebin;
        if (refId[j] >= 0) {
          const auto& trkRef = trackRefs[refId[j]];
          info.x = trkRef.X();
          info.y = trkRef.Y();
          info.z = trkRef.Z();
          info.pX = trkRef.Px();
          info.pY = trkRef.Py();
          info.pZ = trkRef.Pz();
          info.genRadius = std::sqrt(trk.GetStartVertexCoordinatesX() * trk.GetStartVertexCoordinatesX() + trk.GetStartVertexCoordinatesY() * trk.GetStartVertexCoordinatesY() + trk.GetStartVertexCoordinatesZ() * trk.GetStartVertexCoordinatesZ());
        } else {
          info.x = info.y = info.z = info.pX = info.pY = info.pZ = 0;
          info.genRadius = 0;
        }
      }
    }
    if (mTracking && mTracking->GetProcessingSettings().debugLevel) {
      GPUInfo("Finished reading O2 Track MC information (%f seconds)", timer.GetCurrentElapsedTime());
    }
  }
#endif

  if (mConfig.matchMCLabels.size()) {
    unsigned int nFiles = mConfig.matchMCLabels.size();
    std::vector<std::unique_ptr<TFile>> files;
    std::vector<std::vector<std::vector<int>>*> labelsBuffer(nFiles);
    std::vector<std::vector<std::vector<int>>*> effBuffer(nFiles);
    for (unsigned int i = 0; i < nFiles; i++) {
      files.emplace_back(std::make_unique<TFile>(mConfig.matchMCLabels[i]));
      labelsBuffer[i] = (std::vector<std::vector<int>>*)files[i]->Get("mcLabelBuffer");
      effBuffer[i] = (std::vector<std::vector<int>>*)files[i]->Get("mcEffBuffer");
      if (labelsBuffer[i] == nullptr || effBuffer[i] == nullptr) {
        GPUError("Error opening / reading from labels file %u/%s: %p %p", i, mConfig.matchMCLabels[i], (void*)labelsBuffer[i], (void*)effBuffer[i]);
        exit(1);
      }
    }

    mGoodTracks.resize(labelsBuffer[0]->size());
    mGoodHits.resize(labelsBuffer[0]->size());
    for (unsigned int iEvent = 0; iEvent < labelsBuffer[0]->size(); iEvent++) {
      std::vector<bool> labelsOK((*effBuffer[0])[iEvent].size());
      for (unsigned int k = 0; k < (*effBuffer[0])[iEvent].size(); k++) {
        labelsOK[k] = false;
        for (unsigned int l = 0; l < nFiles; l++) {
          if ((*effBuffer[0])[iEvent][k] != (*effBuffer[l])[iEvent][k]) {
            labelsOK[k] = true;
            break;
          }
        }
      }
      mGoodTracks[iEvent].resize((*labelsBuffer[0])[iEvent].size());
      for (unsigned int k = 0; k < (*labelsBuffer[0])[iEvent].size(); k++) {
        if ((*labelsBuffer[0])[iEvent][k] == MC_LABEL_INVALID) {
          continue;
        }
        mGoodTracks[iEvent][k] = labelsOK[abs((*labelsBuffer[0])[iEvent][k])];
      }
    }
  }
  mQAInitialized = true;
  return 0;
}

void GPUQA::RunQA(bool matchOnly, const std::vector<o2::tpc::TrackTPC>* tracksExternal, const std::vector<o2::MCCompLabel>* tracksExtMC, const o2::tpc::ClusterNativeAccess* clNative)
{
  if (!mQAInitialized) {
    throw std::runtime_error("QA not initialized");
  }
  if (!clNative && mTracking) {
    clNative = mTracking->mIOPtrs.clustersNative;
  }
  mClNative = clNative;

  // Initialize Arrays
  unsigned int nReconstructedTracks = 0;
  if (tracksExternal) {
#ifdef GPUCA_O2_LIB
    nReconstructedTracks = tracksExternal->size();
#endif
  } else {
    nReconstructedTracks = mTracking->mIOPtrs.nMergedTracks;
  }
  mTrackMCLabels.resize(nReconstructedTracks);
  for (unsigned int iCol = 0; iCol < GetNMCCollissions(); iCol++) {
    mTrackMCLabelsReverse[iCol].resize(GetNMCTracks(iCol));
    mRecTracks[iCol].resize(GetNMCTracks(iCol));
    mFakeTracks[iCol].resize(GetNMCTracks(iCol));
    mMCParam[iCol].resize(GetNMCTracks(iCol));
    memset(mRecTracks[iCol].data(), 0, mRecTracks[iCol].size() * sizeof(mRecTracks[iCol][0]));
    memset(mFakeTracks[iCol].data(), 0, mFakeTracks[iCol].size() * sizeof(mFakeTracks[iCol][0]));
    for (size_t i = 0; i < mTrackMCLabelsReverse[iCol].size(); i++) {
      mTrackMCLabelsReverse[iCol][i] = -1;
    }
  }
  if (mQATasks & taskClusterAttach) {
    mClusterParam.resize(GetNMCLabels());
    memset(mClusterParam.data(), 0, mClusterParam.size() * sizeof(mClusterParam[0]));
  }
  HighResTimer timer;

  mNEvents++;
  if (mConfig.writeMCLabels) {
    mcEffBuffer.resize(mNEvents);
    mcLabelBuffer.resize(mNEvents);
    mcEffBuffer[mNEvents - 1].resize(GetNMCTracks(0));
    mcLabelBuffer[mNEvents - 1].resize(nReconstructedTracks);
  }

  bool mcAvail = mcPresent() || tracksExtMC;

  if (mcAvail && !tracksExtMC && mTracking->GetParam().rec.NonConsecutiveIDs) {
    GPUError("QA incompatible to non-consecutive MC labels");
    return;
  }

  if (mcAvail) {
    // Assign Track MC Labels
    timer.Start();
    if (tracksExternal) {
#ifdef GPUCA_O2_LIB
      for (unsigned int i = 0; i < tracksExternal->size(); i++) {
        mTrackMCLabels[i] = (*tracksExtMC)[i];
      }
#endif
    } else {
      auto acc = GPUTPCTrkLbl<true, mcLabelI_t>(GetClusterLabels(), 1.f - mConfig.recThreshold);
#if QA_DEBUG == 0
      GPUCA_OPENMP(parallel for firstprivate(acc))
#endif
      for (unsigned int i = 0; i < nReconstructedTracks; i++) {
        acc.reset();
        int nClusters = 0;
        const GPUTPCGMMergedTrack& track = mTracking->mIOPtrs.mergedTracks[i];
        std::vector<mcLabel_t> labels;
        for (unsigned int k = 0; k < track.NClusters(); k++) {
          if (mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject) {
            continue;
          }
          nClusters++;
          unsigned int hitId = mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].num;
          if (hitId >= GetNMCLabels()) {
            GPUError("Invalid hit id %u > %d", hitId, GetNMCLabels());
            throw std::runtime_error("qa error");
          }
          acc.addLabel(hitId);
          for (int j = 0; j < GetMCLabelNID(hitId); j++) {
            if (GetMCLabelID(hitId, j) >= (int)GetNMCTracks(GetMCLabelCol(hitId, j))) {
              GPUError("Invalid label %d > %d (hit %d, label %d, col %d)", GetMCLabelID(hitId, j), GetNMCTracks(GetMCLabelCol(hitId, j)), hitId, j, (int)GetMCLabelCol(hitId, j));
              throw std::runtime_error("qa error");
            }
            if (GetMCLabelID(hitId, j) >= 0) {
              if (QA_DEBUG >= 3 && track.OK()) {
                GPUInfo("Track %d Cluster %u Label %d: %d (%f)", i, k, j, GetMCLabelID(hitId, j), GetMCLabelWeight(hitId, j));
              }
            }
          }
        }

        float maxweight, sumweight;
        int maxcount;
        auto maxLabel = acc.computeLabel(&maxweight, &sumweight, &maxcount);
        mTrackMCLabels[i] = maxLabel;
        if (QA_DEBUG && track.OK() && GetNMCTracks(maxLabel.getEventID()) > (unsigned int)maxLabel.getTrackID()) {
          const mcInfo_t& mc = GetMCTrack(maxLabel);
          GPUInfo("Track %d label %d (fake %d) weight %f clusters %d (fitted %d) (%f%% %f%%) Pt %f", i, maxLabel.getTrackID(), (int)(maxLabel.isFake()), maxweight, nClusters, track.NClustersFitted(), 100.f * maxweight / sumweight, 100.f * (float)maxcount / (float)nClusters,
                  std::sqrt(mc.pX * mc.pX + mc.pY * mc.pY));
        }
      }
    }
    if (QA_TIMING) {
      GPUInfo("QA Time: Assign Track Labels:\t\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
    }

    for (unsigned int i = 0; i < nReconstructedTracks; i++) {
      const GPUTPCGMMergedTrack* track = mTracking ? &mTracking->mIOPtrs.mergedTracks[i] : nullptr;
      mcLabelI_t label = mTrackMCLabels[i];
      if (mQATasks & taskClusterAttach) {
        // fill cluster attachment status
        if (!track->OK()) {
          continue;
        }
        if (!mTrackMCLabels[i].isValid()) {
          for (unsigned int k = 0; k < track->NClusters(); k++) {
            if (mTracking->mIOPtrs.mergedTrackHits[track->FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject) {
              continue;
            }
            mClusterParam[mTracking->mIOPtrs.mergedTrackHits[track->FirstClusterRef() + k].num].fakeAttached++;
          }
          continue;
        }
        if (mMCTrackMin == -1 || (label.getTrackID() >= mMCTrackMin && label.getTrackID() < mMCTrackMax)) {
          for (unsigned int k = 0; k < track->NClusters(); k++) {
            if (mTracking->mIOPtrs.mergedTrackHits[track->FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject) {
              continue;
            }
            int hitId = mTracking->mIOPtrs.mergedTrackHits[track->FirstClusterRef() + k].num;
            bool correct = false;
            for (int j = 0; j < GetMCLabelNID(hitId); j++) {
              if (label == GetMCLabel(hitId, j)) {
                correct = true;
                break;
              }
            }
            if (correct) {
              mClusterParam[hitId].attached++;
            } else {
              mClusterParam[hitId].fakeAttached++;
            }
          }
        }
      }

      if (mTrackMCLabels[i].isFake()) {
        (GetMCTrackObj(mFakeTracks, label))++;
      } else if (tracksExternal || !track->MergedLooper()) {
        GetMCTrackObj(mRecTracks, label)++;
        if (mMCTrackMin == -1 || (label.getTrackID() >= mMCTrackMin && label.getTrackID() < mMCTrackMax)) {
          int& revLabel = GetMCTrackObj(mTrackMCLabelsReverse, label);
          if (tracksExternal) {
#ifdef GPUCA_O2_LIB
            if (revLabel == -1 || fabsf((*tracksExternal)[i].getZ()) < fabsf((*tracksExternal)[revLabel].getZ())) {
              revLabel = i;
            }
#endif
          } else {
            const auto* trks = mTracking->mIOPtrs.mergedTracks;
            bool comp;
            if (revLabel == -1) {
              comp = true;
            } else if (mTracking->GetParam().par.earlyTpcTransform) {
              comp = fabsf(trks[i].GetParam().GetZ() + trks[i].GetParam().GetTZOffset()) < fabsf(trks[revLabel].GetParam().GetZ() + trks[revLabel].GetParam().GetTZOffset());
            } else {
              float shift1 = mTracking->GetTPCTransform()->convDeltaTimeToDeltaZinTimeFrame(trks[i].CSide() * GPUChainTracking::NSLICES / 2, trks[i].GetParam().GetTZOffset());
              float shift2 = mTracking->GetTPCTransform()->convDeltaTimeToDeltaZinTimeFrame(trks[revLabel].CSide() * GPUChainTracking::NSLICES / 2, trks[revLabel].GetParam().GetTZOffset());
              comp = fabsf(trks[i].GetParam().GetZ() + shift1) < fabsf(trks[revLabel].GetParam().GetZ() + shift2);
            }
            if (revLabel == -1 || !trks[revLabel].OK() || (trks[i].OK() && comp)) {
              revLabel = i;
            }
          }
        }
      }
    }
    if (mQATasks & taskClusterAttach) {
      // fill cluster adjacent status
      for (unsigned int i = 0; i < GetNMCLabels(); i++) {
        if (mClusterParam[i].attached == 0 && mClusterParam[i].fakeAttached == 0) {
          int attach = mTracking->mIOPtrs.mergedTrackHitAttachment[i];
          if (attach & gputpcgmmergertypes::attachFlagMask) {
            int track = attach & gputpcgmmergertypes::attachTrackMask;
            mcLabelI_t trackL = mTrackMCLabels[track];
            bool fake = true;
            for (int j = 0; j < GetMCLabelNID(i); j++) {
              // GPUInfo("Attach %x Track %d / %d:%d", attach, track, j, GetMCLabelID(i, j));
              if (trackL == GetMCLabel(i, j)) {
                fake = false;
                break;
              }
            }
            if (fake) {
              mClusterParam[i].fakeAdjacent++;
            } else {
              mClusterParam[i].adjacent++;
            }
          }
        }
      }
    }

    if (mConfig.matchMCLabels.size()) {
      mGoodHits[mNEvents - 1].resize(GetNMCLabels());
      std::vector<bool> allowMCLabels(GetNMCTracks(0));
      for (unsigned int k = 0; k < GetNMCTracks(0); k++) {
        allowMCLabels[k] = false;
      }
      for (unsigned int i = 0; i < nReconstructedTracks; i++) {
        if (!mGoodTracks[mNEvents - 1][i]) {
          continue;
        }
        if (mConfig.matchDisplayMinPt > 0) {
          if (!mTrackMCLabels[i].isValid()) {
            continue;
          }
          const mcInfo_t& info = GetMCTrack(mTrackMCLabels[i]);
          if (info.pX * info.pX + info.pY * info.pY < mConfig.matchDisplayMinPt * mConfig.matchDisplayMinPt) {
            continue;
          }
        }

        const GPUTPCGMMergedTrack& track = mTracking->mIOPtrs.mergedTracks[i];
        for (unsigned int j = 0; j < track.NClusters(); j++) {
          int hitId = mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + j].num;
          if (GetMCLabelNID(hitId)) {
            int mcID = GetMCLabelID(hitId, 0);
            if (mcID >= 0) {
              allowMCLabels[mcID] = true;
            }
          }
        }
      }
      for (unsigned int i = 0; i < GetNMCLabels(); i++) {
        for (int j = 0; j < GetMCLabelNID(i); j++) {
          int mcID = GetMCLabelID(i, j);
          if (mcID >= 0 && allowMCLabels[mcID]) {
            mGoodHits[mNEvents - 1][i] = true;
          }
        }
      }
    }
    if (QA_TIMING) {
      GPUInfo("QA Time: Cluster attach status:\t\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
    }

    if (matchOnly) {
      return;
    }

    // Recompute fNWeightCls (might have changed after merging events into timeframes)
    for (unsigned int iCol = 0; iCol < GetNMCCollissions(); iCol++) {
      for (unsigned int i = 0; i < GetNMCTracks(iCol); i++) {
        mMCParam[iCol][i].nWeightCls = 0.;
      }
    }
    for (unsigned int i = 0; i < GetNMCLabels(); i++) {
      float weightTotal = 0.f;
      for (int j = 0; j < GetMCLabelNID(i); j++) {
        if (GetMCLabelID(i, j) >= 0) {
          weightTotal += GetMCLabelWeight(i, j);
        }
      }
      for (int j = 0; j < GetMCLabelNID(i); j++) {
        if (GetMCLabelID(i, j) >= 0) {
          GetMCTrackObj(mMCParam, GetMCLabel(i, j)).nWeightCls += GetMCLabelWeight(i, j) / weightTotal;
        }
      }
    }
    if (QA_TIMING) {
      GPUInfo("QA Time: Compute cluster label weights:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
    }

    // Compute MC Track Parameters for MC Tracks
    GPUCA_OPENMP(parallel for)
    for (unsigned int iCol = 0; iCol < GetNMCCollissions(); iCol++) {
      for (unsigned int i = 0; i < GetNMCTracks(iCol); i++) {
        const mcInfo_t& info = GetMCTrack(i, iCol);
        additionalMCParameters& mc2 = mMCParam[iCol][i];
        mc2.pt = std::sqrt(info.pX * info.pX + info.pY * info.pY);
        mc2.phi = M_PI + std::atan2(-info.pY, -info.pX);
        float p = info.pX * info.pX + info.pY * info.pY + info.pZ * info.pZ;
        if (p < 1e-18) {
          mc2.theta = mc2.eta = 0.f;
        } else {
          mc2.theta = info.pZ == 0 ? (M_PI / 2) : (std::acos(info.pZ / std::sqrt(p)));
          mc2.eta = -std::log(std::tan(0.5 * mc2.theta));
        }
        if (mConfig.writeMCLabels) {
          std::vector<int>& effBuffer = mcEffBuffer[mNEvents - 1];
          effBuffer[i] = mRecTracks[iCol][i] * 1000 + mFakeTracks[iCol][i];
        }
      }
    }
    if (QA_TIMING) {
      GPUInfo("QA Time: Compute track mc parameters:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
    }

    // Fill Efficiency Histograms
    if (mQATasks & taskTrackingEff) {
      for (unsigned int iCol = 0; iCol < GetNMCCollissions(); iCol++) {
        for (unsigned int i = 0; i < GetNMCTracks(iCol); i++) {
          if ((mMCTrackMin != -1 && (int)i < mMCTrackMin) || (mMCTrackMax != -1 && (int)i >= mMCTrackMax)) {
            continue;
          }
          const mcInfo_t& info = GetMCTrack(i, iCol);
          const additionalMCParameters& mc2 = mMCParam[iCol][i];
          if (mc2.nWeightCls == 0.f) {
            continue;
          }
          const float& mcpt = mc2.pt;
          const float& mcphi = mc2.phi;
          const float& mceta = mc2.eta;

          if (info.primDaughters) {
            continue;
          }
          if (mc2.nWeightCls < MIN_WEIGHT_CLS) {
            continue;
          }
          int findable = mc2.nWeightCls >= FINDABLE_WEIGHT_CLS;
          if (info.pid < 0) {
            continue;
          }
          if (info.charge == 0.f) {
            continue;
          }
          if (mConfig.filterCharge && info.charge * mConfig.filterCharge < 0) {
            continue;
          }
          if (mConfig.filterPID >= 0 && info.pid != mConfig.filterPID) {
            continue;
          }

          if (fabsf(mceta) > ETA_MAX || mcpt < PT_MIN || mcpt > PT_MAX) {
            continue;
          }

          float alpha = std::atan2(info.y, info.x);
          alpha /= M_PI / 9.f;
          alpha = std::floor(alpha);
          alpha *= M_PI / 9.f;
          alpha += M_PI / 18.f;

          float c = std::cos(alpha);
          float s = std::sin(alpha);
          float localY = -info.x * s + info.y * c;

          for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 2; k++) {
              if (k == 0 && findable == 0) {
                continue;
              }

              int val = (j == 0) ? (mRecTracks[iCol][i] ? 1 : 0) : (j == 1) ? (mRecTracks[iCol][i] ? mRecTracks[iCol][i] - 1 : 0) : (j == 2) ? mFakeTracks[iCol][i] : 1;
              if (val == 0) {
                continue;
              }

              for (int l = 0; l < 5; l++) {
                if (info.prim && mcpt < PT_MIN_PRIM) {
                  continue;
                }
                if (l != 3 && fabsf(mceta) > ETA_MAX2) {
                  continue;
                }
                if (l < 4 && mcpt < 1.f / mConfig.qpt) {
                  continue;
                }

                float pos = l == 0 ? localY : l == 1 ? info.z : l == 2 ? mcphi : l == 3 ? mceta : mcpt;

                mEff[j][k][!info.prim][l][0]->Fill(pos, val);
              }
            }
          }
        }
      }
      if (QA_TIMING) {
        GPUInfo("QA Time: Fill efficiency histograms:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
      }
    }

    // Fill Resolution Histograms
    if (mQATasks & (taskTrackingRes | taskTrackingResPull)) {
      GPUTPCGMPropagator prop;
      prop.SetMaxSinPhi(.999);
      prop.SetMaterialTPC();
      prop.SetPolynomialField(&mParam.polynomialField);
      prop.SetToyMCEventsFlag(mParam.par.ToyMCEventsFlag);

      for (unsigned int i = 0; i < mTrackMCLabels.size(); i++) {
        if (mConfig.writeMCLabels) {
          std::vector<int>& labelBuffer = mcLabelBuffer[mNEvents - 1];
          labelBuffer[i] = mTrackMCLabels[i].getTrackID();
        }
        if (mTrackMCLabels[i].isFake()) {
          continue;
        }
        const mcInfo_t& mc1 = GetMCTrack(mTrackMCLabels[i]);
        const additionalMCParameters& mc2 = GetMCTrackObj(mMCParam, mTrackMCLabels[i]);

        if (mc1.primDaughters) {
          continue;
        }
        if (!tracksExternal) {
          if (!mTracking->mIOPtrs.mergedTracks[i].OK()) {
            continue;
          }
          if (mTracking->mIOPtrs.mergedTracks[i].MergedLooper()) {
            continue;
          }
        }
        if ((mMCTrackMin != -1 && mTrackMCLabels[i].getTrackID() < mMCTrackMin) || (mMCTrackMax != -1 && mTrackMCLabels[i].getTrackID() >= mMCTrackMax)) {
          continue;
        }
        if (fabsf(mc2.eta) > ETA_MAX || mc2.pt < PT_MIN || mc2.pt > PT_MAX) {
          continue;
        }
        if (mc1.charge == 0.f) {
          continue;
        }
        if (mc1.pid < 0) {
          continue;
        }
        if (mConfig.filterCharge && mc1.charge * mConfig.filterCharge < 0) {
          continue;
        }
        if (mConfig.filterPID >= 0 && mc1.pid != mConfig.filterPID) {
          continue;
        }
        if (mc2.nWeightCls < MIN_WEIGHT_CLS) {
          continue;
        }
        if (mConfig.resPrimaries == 1 && !mc1.prim) {
          continue;
        } else if (mConfig.resPrimaries == 2 && mc1.prim) {
          continue;
        }
        if (GetMCTrackObj(mTrackMCLabelsReverse, mTrackMCLabels[i]) != (int)i) {
          continue;
        }

        GPUTPCGMTrackParam param;
        float alpha = 0.f;
        int side;
        if (tracksExternal) {
#ifdef GPUCA_O2_LIB
          for (int k = 0; k < 5; k++) {
            param.Par()[k] = (*tracksExternal)[i].getParams()[k];
          }
          for (int k = 0; k < 15; k++) {
            param.Cov()[k] = (*tracksExternal)[i].getCov()[k];
          }
          param.X() = (*tracksExternal)[i].getX();
          param.TZOffset() = (*tracksExternal)[i].getTime0();
          alpha = (*tracksExternal)[i].getAlpha();
          side = (*tracksExternal)[i].hasBothSidesClusters() ? 2 : ((*tracksExternal)[i].hasCSideClusters() ? 1 : 0);
#endif
        } else {
          param = mTracking->mIOPtrs.mergedTracks[i].GetParam();
          alpha = mTracking->mIOPtrs.mergedTracks[i].GetAlpha();
          side = mTracking->mIOPtrs.mergedTracks[i].CCE() ? 2 : (mTracking->mIOPtrs.mergedTracks[i].CSide() ? 1 : 0);
        }

        float mclocal[4]; // Rotated x,y,Px,Py mc-coordinates - the MC data should be rotated since the track is propagated best along x
        float c = std::cos(alpha);
        float s = std::sin(alpha);
        float x = mc1.x;
        float y = mc1.y;
        mclocal[0] = x * c + y * s;
        mclocal[1] = -x * s + y * c;
        float px = mc1.pX;
        float py = mc1.pY;
        mclocal[2] = px * c + py * s;
        mclocal[3] = -px * s + py * c;

        if (mclocal[0] < TRACK_EXPECTED_REFERENCE_X - 3) {
          continue;
        }
        if (mclocal[0] > param.GetX() + 20) {
          continue;
        }
        if (param.GetX() > mConfig.maxResX) {
          continue;
        }

        auto getdz = [this, &param, &mc1, &side, tracksExternal]() {
          if (tracksExternal) {
            return param.GetZ();
          }
          if (!mParam.par.continuousMaxTimeBin) {
            return param.GetZ() - mc1.z;
          }
#ifdef GPUCA_TPC_GEOMETRY_O2
          if (!mParam.par.earlyTpcTransform) {
            float shift = side == 2 ? 0 : mTracking->GetTPCTransform()->convDeltaTimeToDeltaZinTimeFrame(side * GPUChainTracking::NSLICES / 2, param.GetTZOffset() - mc1.t0);
            return param.GetZ() + shift - mc1.z;
          }
#endif
          return param.Z() + param.TZOffset() - mc1.z;
        };

        prop.SetTrack(&param, alpha);
        bool inFlyDirection = 0;
        if (mConfig.strict) {
          const float dx = param.X() - std::max<float>(mclocal[0], TRACK_EXPECTED_REFERENCE_X_DEFAULT); // Limit distance check if the O2 MC position is farther inside than the AliRoot MC position.
          const float dy = param.Y() - mclocal[1];
          const float dz = getdz();
          if (dx * dx + dy * dy + dz * dz > 5.f * 5.f) {
            continue;
          }
        }

        if (prop.PropagateToXAlpha(mclocal[0], alpha, inFlyDirection)) {
          continue;
        }
        if (fabsf(param.Y() - mclocal[1]) > (mConfig.strict ? 1.f : 4.f) || fabsf(getdz()) > (mConfig.strict ? 1.f : 4.f)) {
          continue;
        }
        float charge = mc1.charge > 0 ? 1.f : -1.f;

        float deltaY = param.GetY() - mclocal[1];
        float deltaZ = getdz();
        float deltaPhiNative = param.GetSinPhi() - mclocal[3] / mc2.pt;
        float deltaPhi = std::asin(param.GetSinPhi()) - std::atan2(mclocal[3], mclocal[2]);
        float deltaLambdaNative = param.GetDzDs() - mc1.pZ / mc2.pt;
        float deltaLambda = std::atan(param.GetDzDs()) - std::atan2(mc1.pZ, mc2.pt);
        float deltaPtNative = (param.GetQPt() - charge / mc2.pt) * charge;
        float deltaPt = (fabsf(1.f / param.GetQPt()) - mc2.pt) / mc2.pt;

        float paramval[5] = {mclocal[1], mc1.z, mc2.phi, mc2.eta, mc2.pt};
        float resval[5] = {deltaY, deltaZ, mConfig.nativeFitResolutions ? deltaPhiNative : deltaPhi, mConfig.nativeFitResolutions ? deltaLambdaNative : deltaLambda, mConfig.nativeFitResolutions ? deltaPtNative : deltaPt};
        float pullval[5] = {deltaY / std::sqrt(param.GetErr2Y()), deltaZ / std::sqrt(param.GetErr2Z()), deltaPhiNative / std::sqrt(param.GetErr2SinPhi()), deltaLambdaNative / std::sqrt(param.GetErr2DzDs()), deltaPtNative / std::sqrt(param.GetErr2QPt())};

        for (int j = 0; j < 5; j++) {
          for (int k = 0; k < 5; k++) {
            if (k != 3 && fabsf(mc2.eta) > ETA_MAX2) {
              continue;
            }
            if (k < 4 && mc2.pt < 1.f / mConfig.qpt) {
              continue;
            }
            if (mQATasks & taskTrackingRes) {
              mRes2[j][k]->Fill(resval[j], paramval[k]);
            }
            if (mQATasks & taskTrackingResPull) {
              mPull2[j][k]->Fill(pullval[j], paramval[k]);
            }
          }
        }
      }
      if (QA_TIMING) {
        GPUInfo("QA Time: Fill resolution histograms:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
      }
    }

    if (mQATasks & taskClusterAttach) {
      // Fill cluster histograms
      for (unsigned int iTrk = 0; iTrk < nReconstructedTracks; iTrk++) {
        const GPUTPCGMMergedTrack& track = mTracking->mIOPtrs.mergedTracks[iTrk];
        if (!track.OK()) {
          continue;
        }
        if (!mTrackMCLabels[iTrk].isValid()) {
          for (unsigned int k = 0; k < track.NClusters(); k++) {
            if (mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject) {
              continue;
            }
            int hitId = mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].num;
            float totalWeight = 0.;
            for (int j = 0; j < GetMCLabelNID(hitId); j++) {
              if (GetMCLabelID(hitId, j) >= 0 && GetMCTrackObj(mMCParam, GetMCLabel(hitId, j)).pt > GPUCA_MIN_TRACK_PT_DEFAULT) {
                totalWeight += GetMCLabelWeight(hitId, j);
              }
            }
            int attach = mTracking->mIOPtrs.mergedTrackHitAttachment[hitId];
            CHECK_CLUSTER_STATE_NOCOUNT();
            if (totalWeight > 0) {
              float weight = 1.f / (totalWeight * (mClusterParam[hitId].attached + mClusterParam[hitId].fakeAttached));
              for (int j = 0; j < GetMCLabelNID(hitId); j++) {
                mcLabelI_t label = GetMCLabel(hitId, j);
                if (!label.isFake() && GetMCTrackObj(mMCParam, label).pt > GPUCA_MIN_TRACK_PT_DEFAULT) {
                  float pt = GetMCTrackObj(mMCParam, label).pt;
                  if (pt < PT_MIN_CLUST) {
                    pt = PT_MIN_CLUST;
                  }
                  mClusters[CL_fake]->Fill(pt, GetMCLabelWeight(hitId, j) * weight);
                  mClusters[CL_att_adj]->Fill(pt, GetMCLabelWeight(hitId, j) * weight);
                  if (GetMCTrackObj(mRecTracks, label)) {
                    mClusters[CL_tracks]->Fill(pt, GetMCLabelWeight(hitId, j) * weight);
                  }
                  mClusters[CL_all]->Fill(pt, GetMCLabelWeight(hitId, j) * weight);
                  if (protect || physics) {
                    mClusters[CL_prot]->Fill(pt, GetMCLabelWeight(hitId, j) * weight);
                  }
                  if (physics) {
                    mClusters[CL_physics]->Fill(pt, GetMCLabelWeight(hitId, j) * weight);
                  }
                }
              }
            } else {
              float weight = 1.f / (mClusterParam[hitId].attached + mClusterParam[hitId].fakeAttached);
              mClusters[CL_fake]->Fill(0.f, weight);
              mClusters[CL_att_adj]->Fill(0.f, weight);
              mClusters[CL_all]->Fill(0.f, weight);
              mClusterCounts.nUnaccessible += weight;
              if (protect || physics) {
                mClusters[CL_prot]->Fill(0.f, weight);
              }
              if (physics) {
                mClusters[CL_physics]->Fill(0.f, weight);
              }
            }
          }
          continue;
        }
        mcLabelI_t label = mTrackMCLabels[iTrk];
        if (mMCTrackMin != -1 && (label.getTrackID() < mMCTrackMin || label.getTrackID() >= mMCTrackMax)) {
          continue;
        }
        for (unsigned int k = 0; k < track.NClusters(); k++) {
          if (mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject) {
            continue;
          }
          int hitId = mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].num;
          float pt = GetMCTrackObj(mMCParam, label).pt;
          if (pt < PT_MIN_CLUST) {
            pt = PT_MIN_CLUST;
          }
          float weight = 1.f / (mClusterParam[hitId].attached + mClusterParam[hitId].fakeAttached);
          bool correct = false;
          for (int j = 0; j < GetMCLabelNID(hitId); j++) {
            if (label == GetMCLabel(hitId, j)) {
              correct = true;
              break;
            }
          }
          if (correct) {
            mClusters[CL_attached]->Fill(pt, weight);
            mClusters[CL_tracks]->Fill(pt, weight);
          } else {
            mClusters[CL_fake]->Fill(pt, weight);
          }
          mClusters[CL_att_adj]->Fill(pt, weight);
          mClusters[CL_all]->Fill(pt, weight);
          int attach = mTracking->mIOPtrs.mergedTrackHitAttachment[hitId];
          CHECK_CLUSTER_STATE_NOCOUNT();
          if (protect || physics) {
            mClusters[CL_prot]->Fill(pt, weight);
          }
          if (physics) {
            mClusters[CL_physics]->Fill(pt, weight);
          }
        }
      }
      for (unsigned int i = 0; i < GetNMCLabels(); i++) {
        if ((mMCTrackMin != -1 && GetMCLabelID(i, 0) < mMCTrackMin) || (mMCTrackMax != -1 && GetMCLabelID(i, 0) >= mMCTrackMax)) {
          continue;
        }
        if (mClusterParam[i].attached || mClusterParam[i].fakeAttached) {
          continue;
        }
        int attach = mTracking->mIOPtrs.mergedTrackHitAttachment[i];
        CHECK_CLUSTER_STATE_NOCOUNT();
        if (mClusterParam[i].adjacent) {
          int label = mTracking->mIOPtrs.mergedTrackHitAttachment[i] & gputpcgmmergertypes::attachTrackMask;
          if (!mTrackMCLabels[label].isValid()) {
            float totalWeight = 0.;
            for (int j = 0; j < GetMCLabelNID(i); j++) {
              mcLabelI_t labelT = GetMCLabel(i, j);
              if (!labelT.isFake() && GetMCTrackObj(mMCParam, labelT).pt > GPUCA_MIN_TRACK_PT_DEFAULT) {
                totalWeight += GetMCLabelWeight(i, j);
              }
            }
            float weight = 1.f / totalWeight;
            if (totalWeight > 0) {
              for (int j = 0; j < GetMCLabelNID(i); j++) {
                mcLabelI_t labelT = GetMCLabel(i, j);
                if (!labelT.isFake() && GetMCTrackObj(mMCParam, labelT).pt > GPUCA_MIN_TRACK_PT_DEFAULT) {
                  float pt = GetMCTrackObj(mMCParam, labelT).pt;
                  if (pt < PT_MIN_CLUST) {
                    pt = PT_MIN_CLUST;
                  }
                  if (GetMCTrackObj(mRecTracks, labelT)) {
                    mClusters[CL_tracks]->Fill(pt, GetMCLabelWeight(i, j) * weight);
                  }
                  mClusters[CL_att_adj]->Fill(pt, GetMCLabelWeight(i, j) * weight);
                  mClusters[CL_fakeAdj]->Fill(pt, GetMCLabelWeight(i, j) * weight);
                  mClusters[CL_all]->Fill(pt, GetMCLabelWeight(i, j) * weight);
                  if (protect || physics) {
                    mClusters[CL_prot]->Fill(pt, GetMCLabelWeight(i, j) * weight);
                  }
                  if (physics) {
                    mClusters[CL_physics]->Fill(pt, GetMCLabelWeight(i, j) * weight);
                  }
                }
              }
            } else {
              mClusters[CL_att_adj]->Fill(0.f, 1.f);
              mClusters[CL_fakeAdj]->Fill(0.f, 1.f);
              mClusters[CL_all]->Fill(0.f, 1.f);
              mClusterCounts.nUnaccessible++;
              if (protect || physics) {
                mClusters[CL_prot]->Fill(0.f, 1.f);
              }
              if (physics) {
                mClusters[CL_physics]->Fill(0.f, 1.f);
              }
            }
          } else {
            float pt = GetMCTrackObj(mMCParam, mTrackMCLabels[label]).pt;
            if (pt < PT_MIN_CLUST) {
              pt = PT_MIN_CLUST;
            }
            mClusters[CL_att_adj]->Fill(pt, 1.f);
            mClusters[CL_tracks]->Fill(pt, 1.f);
            mClusters[CL_all]->Fill(pt, 1.f);
            if (protect || physics) {
              mClusters[CL_prot]->Fill(pt, 1.f);
            }
            if (physics) {
              mClusters[CL_physics]->Fill(pt, 1.f);
            }
          }
        } else {
          float totalWeight = 0.;
          for (int j = 0; j < GetMCLabelNID(i); j++) {
            mcLabelI_t labelT = GetMCLabel(i, j);
            if (!labelT.isFake() && GetMCTrackObj(mMCParam, labelT).pt > GPUCA_MIN_TRACK_PT_DEFAULT) {
              totalWeight += GetMCLabelWeight(i, j);
            }
          }
          if (totalWeight > 0) {
            for (int j = 0; j < GetMCLabelNID(i); j++) {
              mcLabelI_t label = GetMCLabel(i, j);
              if (!label.isFake() && GetMCTrackObj(mMCParam, label).pt > GPUCA_MIN_TRACK_PT_DEFAULT) {
                float pt = GetMCTrackObj(mMCParam, label).pt;
                if (pt < PT_MIN_CLUST) {
                  pt = PT_MIN_CLUST;
                }
                float weight = GetMCLabelWeight(i, j) / totalWeight;
                if (mClusterParam[i].fakeAdjacent) {
                  mClusters[CL_fakeAdj]->Fill(pt, weight);
                }
                if (mClusterParam[i].fakeAdjacent) {
                  mClusters[CL_att_adj]->Fill(pt, weight);
                }
                if (GetMCTrackObj(mRecTracks, label)) {
                  mClusters[CL_tracks]->Fill(pt, weight);
                }
                mClusters[CL_all]->Fill(pt, weight);
                if (protect || physics) {
                  mClusters[CL_prot]->Fill(pt, weight);
                }
                if (physics) {
                  mClusters[CL_physics]->Fill(pt, weight);
                }
              }
            }
          } else {
            if (mClusterParam[i].fakeAdjacent) {
              mClusters[CL_fakeAdj]->Fill(0.f, 1.f);
            }
            if (mClusterParam[i].fakeAdjacent) {
              mClusters[CL_att_adj]->Fill(0.f, 1.f);
            }
            mClusters[CL_all]->Fill(0.f, 1.f);
            mClusterCounts.nUnaccessible++;
            if (protect || physics) {
              mClusters[CL_prot]->Fill(0.f, 1.f);
            }
            if (physics) {
              mClusters[CL_physics]->Fill(0.f, 1.f);
            }
          }
        }
      }

      if (QA_TIMING) {
        GPUInfo("QA Time: Fill cluster histograms:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
      }
    }
  } else if (!mConfig.inputHistogramsOnly && !mConfig.noMC && (mQATasks & (taskTrackingEff | taskTrackingRes | taskTrackingResPull | taskClusterAttach))) {
    GPUWarning("No MC information available, only running partial TPC QA!");
  }

  if (mQATasks & taskTrackStatistics) {
    // Fill track statistic histograms
    for (unsigned int i = 0; i < nReconstructedTracks; i++) {
      const GPUTPCGMMergedTrack& track = mTracking->mIOPtrs.mergedTracks[i];
      if (!track.OK()) {
        continue;
      }
      mTracks->Fill(1.f / fabsf(track.GetParam().GetQPt()));
      mNCl->Fill(track.NClustersFitted());
    }

    if (QA_TIMING) {
      GPUInfo("QA Time: Fill track statistics:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
    }
  }

  unsigned int nCl = clNative ? clNative->nClustersTotal : mTracking->GetTPCMerger().NMaxClusters();
  mClusterCounts.nTotal += nCl;
  if (mQATasks & taskClusterCounts) {
    for (unsigned int i = 0; i < nCl; i++) {
      int attach = mTracking->mIOPtrs.mergedTrackHitAttachment[i];
      CHECK_CLUSTER_STATE();

      if (mcAvail) {
        float totalWeight = 0, weight400 = 0, weight40 = 0;
        for (int j = 0; j < GetMCLabelNID(i); j++) {
          const auto& label = GetMCLabel(i, j);
          if (GetMCLabelID(label) >= 0) {
            totalWeight += GetMCLabelWeight(label);
            if (GetMCTrackObj(mMCParam, label).pt >= 0.4) {
              weight400 += GetMCLabelWeight(label);
            }
            if (GetMCTrackObj(mMCParam, label).pt <= 0.04) {
              weight40 += GetMCLabelWeight(label);
            }
          }
        }
        if (totalWeight > 0 && 10.f * weight400 >= totalWeight) {
          if (!unattached && !protect && !physics) {
            mClusterCounts.nFakeRemove400++;
            int totalFake = weight400 < 0.9f * totalWeight;
            if (totalFake) {
              mClusterCounts.nFullFakeRemove400++;
            }
            /*printf("Fake removal (%d): Hit %7d, attached %d lowPt %d looper %d tube200 %d highIncl %d tube %d bad %d recPt %7.2f recLabel %6d", totalFake, i, (int) (mClusterParam[i].attached || mClusterParam[i].fakeAttached),
                (int) lowPt, (int) ((attach & gputpcgmmergertypes::attachGoodLeg) == 0), (int) ((attach & gputpcgmmergertypes::attachTube) && mev200),
                (int) ((attach & gputpcgmmergertypes::attachHighIncl) != 0), (int) ((attach & gputpcgmmergertypes::attachTube) != 0), (int) ((attach & gputpcgmmergertypes::attachGood) == 0),
                fabsf(qpt) > 0 ? 1.f / qpt : 0.f, id);
            for (int j = 0;j < GetMCLabelNID(i);j++)
            {
                //if (GetMCLabelID(i, j) < 0) break;
                printf(" - label%d %6d weight %5d", j, GetMCLabelID(i, j), (int) GetMCLabelWeight(i, j));
                if (GetMCLabelID(i, j) >= 0) printf(" - pt %7.2f", mMCParam[GetMCLabelID(i, j)].pt);
                else printf("             ");
            }
            printf("\n");*/
          }
          mClusterCounts.nAbove400++;
        }
        if (totalWeight > 0 && weight40 >= 0.9 * totalWeight) {
          mClusterCounts.nBelow40++;
          if (protect || physics) {
            mClusterCounts.nFakeProtect40++;
          }
        }
      }
      if (physics) {
        mClusterCounts.nPhysics++;
      }
      if (physics || protect) {
        mClusterCounts.nProt++;
      }
      if (unattached) {
        mClusterCounts.nUnattached++;
      }
    }
  }

  // Process cluster count statistics
  if ((mQATasks & taskClusterCounts) && mConfig.clusterRejectionHistograms) {
    DoClusterCounts(nullptr);
    mClusterCounts = counts_t();
  }

  if (QA_TIMING) {
    GPUInfo("QA Time: Cluster Counts:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
  }

  // Create CSV DumpTrackHits
  if (mConfig.csvDump) {
    if (!mTracking->GetParam().par.earlyTpcTransform) {
      GPUError("Unsupported settings for csv dump\n");
      return;
    }
    int totalNCls = GetNMCLabels();
    if (totalNCls == 0) {
      for (unsigned int iSlice = 0; iSlice < GPUChainTracking::NSLICES; iSlice++) {
        totalNCls += mTracking->mIOPtrs.nClusterData[iSlice];
      }
    }

    std::vector<float> clusterInfo(totalNCls);
    memset(clusterInfo.data(), 0, clusterInfo.size() * sizeof(clusterInfo[0]));
    for (unsigned int i = 0; i < nReconstructedTracks; i++) {
      const GPUTPCGMMergedTrack& track = mTracking->mIOPtrs.mergedTracks[i];
      if (!track.OK()) {
        continue;
      }
      for (unsigned int k = 0; k < track.NClusters(); k++) {
        if (mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject) {
          continue;
        }
        int hitId = mTracking->mIOPtrs.mergedTrackHits[track.FirstClusterRef() + k].num;
        float pt = fabsf(1.f / track.GetParam().GetQPt());
        if (pt > clusterInfo[hitId]) {
          clusterInfo[hitId] = pt;
        }
      }
    }
    static int csvNum = 0;
    char fname[256];
    sprintf(fname, "dump.%d.csv", csvNum);
    FILE* fp = fopen(fname, "w+");
    fprintf(fp, "x;y;z;reconstructedPt;individualMomentum;individualTransverseMomentum;trackLabel1;trackLabel2;trackLabel3;removed\n\n");
    int dumpClTot = 0, dumpClLeft = 0, dumpClRem = 0;
    for (unsigned int iSlice = 0; iSlice < GPUChainTracking::NSLICES; iSlice++) {
      for (unsigned int i = 0; i < mTracking->mIOPtrs.nClusterData[iSlice]; i++) {
        const auto& cl = mTracking->mIOPtrs.clusterData[iSlice][i];
        float x, y, z;
        const int cid = cl.id;
        mTracking->GetParam().Slice2Global(iSlice, cl.x, cl.y, cl.z, &x, &y, &z);

        float totalWeight = 0.f;
        if (mcPresent()) {
          for (int j = 0; j < GetMCLabelNID(cid); j++) {
            if (GetMCLabelID(cid, j) >= 0) {
              totalWeight += GetMCLabelWeight(cid, j);
            }
          }
        }

        float maxPt = 0.;
        float p = 0.;

        if (totalWeight > 0) {
          for (int j = 0; j < GetMCLabelNID(cid); j++) {
            const mcLabelI_t label = GetMCLabel(cid, j);
            if (!label.isFake() && GetMCLabelWeight(cid, j) > 0.3 * totalWeight) {
              const mcInfo_t& info = GetMCTrack(label);
              const additionalMCParameters& mc2 = GetMCTrackObj(mMCParam, label);
              const float pt = fabsf(mc2.pt);
              if (pt > maxPt) {
                maxPt = pt;
                p = std::sqrt(info.pX * info.pX + info.pY * info.pY + info.pZ * info.pZ);
              }
            }
          }
        }
        int labels[3] = {};
        if (mcPresent()) {
          for (int j = 0; j < GetMCLabelNID(cid); j++) {
            labels[j] = GetMCLabelID(cid, j);
          }
        }

        dumpClTot++;
        int attach = mTracking->mIOPtrs.mergedTrackHitAttachment[cid];
        CHECK_CLUSTER_STATE();
        if (protect || physics) {
          continue;
        }
        if (attach && qpt < 50) {
          continue;
        }
        dumpClLeft++;
        if (attach) {
          dumpClRem++;
        }

        fprintf(fp, "%f;%f;%f;%f;%f;%f;%d;%d;%d;%d\n", x, y, z, attach ? 1.f / qpt : 0.f, p, maxPt, labels[0], labels[1], labels[2], attach ? 1 : 0);
      }
    }
    fclose(fp);
    if (mcPresent()) {
      sprintf(fname, "dump_event.%d.csv", csvNum++);
      fp = fopen(fname, "w+");
      fprintf(fp, "trackLabel;trackMomentum;trackMomentumTransverse;trackMomentumZ\n\n");
      for (unsigned int iCol = 0; iCol < GetNMCCollissions(); iCol++) {
        for (unsigned int i = 0; i < GetNMCTracks(iCol); i++) {
          const mcInfo_t& info = GetMCTrack(i, iCol);
          additionalMCParameters& mc2 = mMCParam[iCol][i];
          if (mc2.nWeightCls > 0) {
            fprintf(fp, "%u;%f;%f;%f\n", i, std::sqrt(info.pX * info.pX + info.pY * info.pY + info.pZ * info.pZ), mc2.pt, info.pZ);
          }
        }
      }
      fclose(fp);
    }
    GPUInfo("Wrote %s,%d clusters in total, %d left, %d to be removed", fname, dumpClTot, dumpClLeft, dumpClRem);
  }
}

void GPUQA::GetName(char* fname, int k)
{
  const int nNewInput = mConfig.inputHistogramsOnly ? 0 : 1;
  if (k || mConfig.inputHistogramsOnly || mConfig.name.size()) {
    if (!(mConfig.inputHistogramsOnly || k)) {
      snprintf(fname, 1024, "%s - ", mConfig.name.c_str());
    } else if (mConfig.compareInputNames.size() > (unsigned)(k - nNewInput)) {
      snprintf(fname, 1024, "%s - ", mConfig.compareInputNames[k - nNewInput]);
    } else {
      strcpy(fname, mConfig.compareInputs[k - nNewInput]);
      if (strlen(fname) > 5 && strcmp(fname + strlen(fname) - 5, ".root") == 0) {
        fname[strlen(fname) - 5] = 0;
      }
      strcat(fname, " - ");
    }
  } else {
    fname[0] = 0;
  }
}

template <class T>
T* GPUQA::GetHist(T*& ee, std::vector<std::unique_ptr<TFile>>& tin, int k, int nNewInput)
{
  T* e = ee;
  if ((mConfig.inputHistogramsOnly || k) && (e = dynamic_cast<T*>(tin[k - nNewInput]->Get(e->GetName()))) == nullptr) {
    GPUWarning("Missing histogram in input %s: %s", mConfig.compareInputs[k - nNewInput], ee->GetName());
    return (nullptr);
  }
  ee = e;
  return (e);
}

void GPUQA::DrawQAHistogramsCleanup()
{
  clearGarbagageCollector();
}

void GPUQA::resetHists()
{
  if (!mQAInitialized) {
    throw std::runtime_error("QA not initialized");
  }
  if (mHaveExternalHists) {
    throw std::runtime_error("Cannot reset external hists");
  }
  for (auto& h : *mHist1D) {
    h.Reset();
  }
  for (auto& h : *mHist2D) {
    h.Reset();
  }
  for (auto& h : *mHist1Dd) {
    h.Reset();
  }
  mClusterCounts = counts_t();
}

int GPUQA::DrawQAHistograms(TObjArray* qcout)
{
  if (!mQAInitialized) {
    throw std::runtime_error("QA not initialized");
  }

  std::vector<Color_t> colorNums(COLORCOUNT);
  for (int i = 0; i < COLORCOUNT; i++) {
    colorNums[i] = qcout ? defaultColorNUms[i] : mColors[i]->GetNumber();
  }

  bool mcAvail = mcPresent();
  char name[2048], fname[1024];

  const int nNewInput = mConfig.inputHistogramsOnly ? 0 : 1;
  const int ConfigNumInputs = nNewInput + mConfig.compareInputs.size();

  std::vector<std::unique_ptr<TFile>> tin;
  for (unsigned int i = 0; i < mConfig.compareInputs.size(); i++) {
    tin.emplace_back(std::make_unique<TFile>(mConfig.compareInputs[i]));
  }
  std::unique_ptr<TFile> tout = nullptr;
  if (mConfig.output.size()) {
    tout = std::make_unique<TFile>(mConfig.output.c_str(), "RECREATE");
  }

  if (mConfig.enableLocalOutput || mConfig.shipToQCAsCanvas) {
    float legendSpacingString = 0.025;
    for (int i = 0; i < ConfigNumInputs; i++) {
      GetName(fname, i);
      if (strlen(fname) * 0.006 > legendSpacingString) {
        legendSpacingString = strlen(fname) * 0.006;
      }
    }

    // Create Canvas / Pads for Efficiency Histograms
    if (mQATasks & taskTrackingEff) {
      for (int ii = 0; ii < 6; ii++) {
        int i = ii == 5 ? 4 : ii;
        sprintf(fname, "eff_vs_%s_layout", VSPARAMETER_NAMES[ii]);
        sprintf(name, "Efficiency versus %s", VSPARAMETER_NAMES[i]);
        mCEff[ii] = createGarbageCollected<TCanvas>(fname, name, 0, 0, 700, 700. * 2. / 3.);
        mCEff[ii]->cd();
        float dy = 1. / 2.;
        mPEff[ii][0] = createGarbageCollected<TPad>("p0", "", 0.0, dy * 0, 0.5, dy * 1);
        mPEff[ii][0]->Draw();
        mPEff[ii][0]->SetRightMargin(0.04);
        mPEff[ii][1] = createGarbageCollected<TPad>("p1", "", 0.5, dy * 0, 1.0, dy * 1);
        mPEff[ii][1]->Draw();
        mPEff[ii][1]->SetRightMargin(0.04);
        mPEff[ii][2] = createGarbageCollected<TPad>("p2", "", 0.0, dy * 1, 0.5, dy * 2 - .001);
        mPEff[ii][2]->Draw();
        mPEff[ii][2]->SetRightMargin(0.04);
        mPEff[ii][3] = createGarbageCollected<TPad>("p3", "", 0.5, dy * 1, 1.0, dy * 2 - .001);
        mPEff[ii][3]->Draw();
        mPEff[ii][3]->SetRightMargin(0.04);
        mLEff[ii] = createGarbageCollected<TLegend>(0.92 - legendSpacingString * 1.45, 0.83 - (0.93 - 0.82) / 2. * (float)ConfigNumInputs, 0.98, 0.849);
        SetLegend(mLEff[ii]);
      }
    }

    // Create Canvas / Pads for Resolution Histograms
    if (mQATasks & taskTrackingRes) {
      for (int ii = 0; ii < 7; ii++) {
        int i = ii == 5 ? 4 : ii;
        if (ii == 6) {
          sprintf(fname, "res_integral_layout");
          sprintf(name, "Integral Resolution");
        } else {
          sprintf(fname, "res_vs_%s_layout", VSPARAMETER_NAMES[ii]);
          sprintf(name, "Resolution versus %s", VSPARAMETER_NAMES[i]);
        }
        mCRes[ii] = createGarbageCollected<TCanvas>(fname, name, 0, 0, 700, 700. * 2. / 3.);
        mCRes[ii]->cd();
        gStyle->SetOptFit(1);

        float dy = 1. / 2.;
        mPRes[ii][3] = createGarbageCollected<TPad>("p0", "", 0.0, dy * 0, 0.5, dy * 1);
        mPRes[ii][3]->Draw();
        mPRes[ii][3]->SetRightMargin(0.04);
        mPRes[ii][4] = createGarbageCollected<TPad>("p1", "", 0.5, dy * 0, 1.0, dy * 1);
        mPRes[ii][4]->Draw();
        mPRes[ii][4]->SetRightMargin(0.04);
        mPRes[ii][0] = createGarbageCollected<TPad>("p2", "", 0.0, dy * 1, 1. / 3., dy * 2 - .001);
        mPRes[ii][0]->Draw();
        mPRes[ii][0]->SetRightMargin(0.04);
        mPRes[ii][0]->SetLeftMargin(0.15);
        mPRes[ii][1] = createGarbageCollected<TPad>("p3", "", 1. / 3., dy * 1, 2. / 3., dy * 2 - .001);
        mPRes[ii][1]->Draw();
        mPRes[ii][1]->SetRightMargin(0.04);
        mPRes[ii][1]->SetLeftMargin(0.135);
        mPRes[ii][2] = createGarbageCollected<TPad>("p4", "", 2. / 3., dy * 1, 1.0, dy * 2 - .001);
        mPRes[ii][2]->Draw();
        mPRes[ii][2]->SetRightMargin(0.06);
        mPRes[ii][2]->SetLeftMargin(0.135);
        if (ii < 6) {
          mLRes[ii] = createGarbageCollected<TLegend>(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.86) / 2. * (float)ConfigNumInputs, 0.98, 0.949);
          SetLegend(mLRes[ii]);
        }
      }
    }

    // Create Canvas / Pads for Pull Histograms
    if (mQATasks & taskTrackingResPull) {
      for (int ii = 0; ii < 7; ii++) {
        int i = ii == 5 ? 4 : ii;

        if (ii == 6) {
          sprintf(fname, "pull_integral_layout");
          sprintf(name, "Integral Pull");
        } else {
          sprintf(fname, "pull_vs_%s_layout", VSPARAMETER_NAMES[ii]);
          sprintf(name, "Pull versus %s", VSPARAMETER_NAMES[i]);
        }
        mCPull[ii] = createGarbageCollected<TCanvas>(fname, name, 0, 0, 700, 700. * 2. / 3.);
        mCPull[ii]->cd();
        gStyle->SetOptFit(1);

        float dy = 1. / 2.;
        mPPull[ii][3] = createGarbageCollected<TPad>("p0", "", 0.0, dy * 0, 0.5, dy * 1);
        mPPull[ii][3]->Draw();
        mPPull[ii][3]->SetRightMargin(0.04);
        mPPull[ii][4] = createGarbageCollected<TPad>("p1", "", 0.5, dy * 0, 1.0, dy * 1);
        mPPull[ii][4]->Draw();
        mPPull[ii][4]->SetRightMargin(0.04);
        mPPull[ii][0] = createGarbageCollected<TPad>("p2", "", 0.0, dy * 1, 1. / 3., dy * 2 - .001);
        mPPull[ii][0]->Draw();
        mPPull[ii][0]->SetRightMargin(0.04);
        mPPull[ii][0]->SetLeftMargin(0.15);
        mPPull[ii][1] = createGarbageCollected<TPad>("p3", "", 1. / 3., dy * 1, 2. / 3., dy * 2 - .001);
        mPPull[ii][1]->Draw();
        mPPull[ii][1]->SetRightMargin(0.04);
        mPPull[ii][1]->SetLeftMargin(0.135);
        mPPull[ii][2] = createGarbageCollected<TPad>("p4", "", 2. / 3., dy * 1, 1.0, dy * 2 - .001);
        mPPull[ii][2]->Draw();
        mPPull[ii][2]->SetRightMargin(0.06);
        mPPull[ii][2]->SetLeftMargin(0.135);
        if (ii < 6) {
          mLPull[ii] = createGarbageCollected<TLegend>(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.86) / 2. * (float)ConfigNumInputs, 0.98, 0.949);
          SetLegend(mLPull[ii]);
        }
      }
    }

    // Create Canvas for Cluster Histos
    if (mQATasks & taskClusterAttach) {
      for (int i = 0; i < 3; i++) {
        sprintf(fname, "clusters_%s_layout", CLUSTER_TYPES[i]);
        mCClust[i] = createGarbageCollected<TCanvas>(fname, CLUSTER_TITLES[i], 0, 0, 700, 700. * 2. / 3.);
        mCClust[i]->cd();
        mPClust[i] = createGarbageCollected<TPad>("p0", "", 0.0, 0.0, 1.0, 1.0);
        mPClust[i]->Draw();
        float y1 = i != 1 ? 0.77 : 0.27, y2 = i != 1 ? 0.9 : 0.42;
        mLClust[i] = createGarbageCollected<TLegend>(i == 2 ? 0.1 : (0.65 - legendSpacingString * 1.45), y2 - (y2 - y1) * (ConfigNumInputs + (i != 1) / 2.) + 0.005, i == 2 ? (0.3 + legendSpacingString * 1.45) : 0.9, y2);
        SetLegend(mLClust[i]);
      }
    }

    // Create Canvas for track statistic histos
    if (mQATasks & taskTrackStatistics) {
      mCTracks = createGarbageCollected<TCanvas>("ctracks", "Track Pt", 0, 0, 700, 700. * 2. / 3.);
      mCTracks->cd();
      mPTracks = createGarbageCollected<TPad>("p0", "", 0.0, 0.0, 1.0, 1.0);
      mPTracks->Draw();
      mLTracks = createGarbageCollected<TLegend>(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.86) / 2. * (float)ConfigNumInputs, 0.98, 0.949);
      SetLegend(mLTracks);

      mCNCl = createGarbageCollected<TCanvas>("cncl", "Number of clusters per track", 0, 0, 700, 700. * 2. / 3.);
      mCNCl->cd();
      mPNCl = createGarbageCollected<TPad>("p0", "", 0.0, 0.0, 1.0, 1.0);
      mPNCl->Draw();
      mLNCl = createGarbageCollected<TLegend>(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.86) / 2. * (float)ConfigNumInputs, 0.98, 0.949);
      SetLegend(mLNCl);
    }
  }

  if (mConfig.enableLocalOutput && !mConfig.inputHistogramsOnly && (mQATasks & taskTrackingEff)) {
    GPUInfo("QA Stats: Eff: Tracks Prim %d (Eta %d, Pt %d) %f%% (%f%%) Sec %d (Eta %d, Pt %d) %f%% (%f%%) -  Res: Tracks %d (Eta %d, Pt %d)", (int)mEff[3][1][0][0][0]->GetEntries(), (int)mEff[3][1][0][3][0]->GetEntries(), (int)mEff[3][1][0][4][0]->GetEntries(),
            mEff[0][0][0][0][0]->GetSumOfWeights() / std::max(1., mEff[3][0][0][0][0]->GetSumOfWeights()), mEff[0][1][0][0][0]->GetSumOfWeights() / std::max(1., mEff[3][1][0][0][0]->GetSumOfWeights()), (int)mEff[3][1][1][0][0]->GetEntries(), (int)mEff[3][1][1][3][0]->GetEntries(),
            (int)mEff[3][1][1][4][0]->GetEntries(), mEff[0][0][1][0][0]->GetSumOfWeights() / std::max(1., mEff[3][0][1][0][0]->GetSumOfWeights()), mEff[0][1][1][0][0]->GetSumOfWeights() / std::max(1., mEff[3][1][1][0][0]->GetSumOfWeights()), (int)mRes2[0][0]->GetEntries(),
            (int)mRes2[0][3]->GetEntries(), (int)mRes2[0][4]->GetEntries());
  }

  int flagShowVsPtLog = (mConfig.enableLocalOutput || mConfig.shipToQCAsCanvas) ? 1 : 0;

  if (mQATasks & taskTrackingEff) {
    // Process / Draw Efficiency Histograms
    for (int ii = 0; ii < 5 + flagShowVsPtLog; ii++) {
      int i = ii == 5 ? 4 : ii;
      for (int k = 0; k < ConfigNumInputs; k++) {
        for (int j = 0; j < 4; j++) {
          if (mConfig.enableLocalOutput || mConfig.shipToQCAsCanvas) {
            mPEff[ii][j]->cd();
          }
          for (int l = 0; l < 3; l++) {
            if (k == 0 && mConfig.inputHistogramsOnly == 0 && ii != 5) {
              if (l == 0) {
                // Divide eff, compute all for fake/clone
                mEff[0][j / 2][j % 2][i][1]->Divide(mEff[l][j / 2][j % 2][i][0], mEff[3][j / 2][j % 2][i][0], 1, 1, "B");
                mEff[3][j / 2][j % 2][i][1]->Reset(); // Sum up rec + clone + fake for clone/fake rate
                mEff[3][j / 2][j % 2][i][1]->Add(mEff[0][j / 2][j % 2][i][0]);
                mEff[3][j / 2][j % 2][i][1]->Add(mEff[1][j / 2][j % 2][i][0]);
                mEff[3][j / 2][j % 2][i][1]->Add(mEff[2][j / 2][j % 2][i][0]);
              } else {
                // Divide fake/clone
                mEff[l][j / 2][j % 2][i][1]->Divide(mEff[l][j / 2][j % 2][i][0], mEff[3][j / 2][j % 2][i][1], 1, 1, "B");
              }
            }

            TH1F* e = mEff[l][j / 2][j % 2][i][1];

            e->SetStats(kFALSE);
            e->SetMaximum(1.02);
            e->SetMinimum(-0.02);
            if (!mConfig.inputHistogramsOnly && k == 0) {
              if (tout) {
                mEff[l][j / 2][j % 2][i][0]->Write();
                e->Write();
                if (l == 2) {
                  mEff[3][j / 2][j % 2][i][0]->Write(); // Store also all histogram!
                }
              }
            } else if (GetHist(e, tin, k, nNewInput) == nullptr) {
              continue;
            }
            e->SetTitle(EFFICIENCY_TITLES[j]);
            e->GetYaxis()->SetTitle("(Efficiency)");
            e->GetXaxis()->SetTitle(XAXIS_TITLES[i]);

            e->SetLineWidth(1);
            e->SetLineStyle(CONFIG_DASHED_MARKERS ? k + 1 : 1);
            SetAxisSize(e);
            if (qcout && !mConfig.shipToQCAsCanvas) {
              qcout->Add(e);
            }
            if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
              continue;
            }
            e->SetMarkerColor(kBlack);
            e->SetLineColor(colorNums[(l == 2 ? (ConfigNumInputs * 2 + k) : (k * 2 + l)) % COLORCOUNT]);
            e->Draw(k || l ? "same" : "");
            if (j == 0) {
              GetName(fname, k);
              sprintf(name, "%s%s", fname, EFF_NAMES[l]);
              mLEff[ii]->AddEntry(e, name, "l");
            }
            if (ii == 5) {
              mPEff[ii][j]->SetLogx();
            }
          }
          if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
            continue;
          }
          mCEff[ii]->cd();
          ChangePadTitleSize(mPEff[ii][j], 0.056);
        }
      }
      if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
        continue;
      }

      mLEff[ii]->Draw();

      if (qcout) {
        qcout->Add(mCEff[ii]);
      }
      if (!mConfig.enableLocalOutput) {
        continue;
      }
      doPerfFigure(0.2, 0.295, 0.025);
      mCEff[ii]->Print(Form("plots/eff_vs_%s.pdf", VSPARAMETER_NAMES[ii]));
      if (mConfig.writeRootFiles) {
        mCEff[ii]->Print(Form("plots/eff_vs_%s.root", VSPARAMETER_NAMES[ii]));
      }
    }
  }

  if (mQATasks & (taskTrackingRes | taskTrackingResPull)) {
    // Process / Draw Resolution Histograms
    TH1D *resIntegral[5] = {}, *pullIntegral[5] = {};
    TCanvas* cfit = nullptr;
    std::unique_ptr<TF1> customGaus = std::make_unique<TF1>("G", "[0]*exp(-(x-[1])*(x-[1])/(2.*[2]*[2]))");
    for (int p = 0; p < 2; p++) {
      if ((p == 0 && (mQATasks & taskTrackingRes) == 0) || (p == 1 && (mQATasks & taskTrackingResPull) == 0)) {
        continue;
      }
      for (int ii = 0; ii < 5 + flagShowVsPtLog; ii++) {
        TCanvas* can = p ? mCPull[ii] : mCRes[ii];
        TLegend* leg = p ? mLPull[ii] : mLRes[ii];
        int i = ii == 5 ? 4 : ii;
        for (int j = 0; j < 5; j++) {
          TH2F* src = p ? mPull2[j][i] : mRes2[j][i];
          TH1F** dst = p ? mPull[j][i] : mRes[j][i];
          TH1D*& dstIntegral = p ? pullIntegral[j] : resIntegral[j];
          TPad* pad = p ? mPPull[ii][j] : mPRes[ii][j];

          if (!mConfig.inputHistogramsOnly && ii != 5) {
            if (cfit == nullptr) {
              cfit = createGarbageCollected<TCanvas>();
            }
            cfit->cd();

            TAxis* axis = src->GetYaxis();
            int nBins = axis->GetNbins();
            int integ = 1;
            for (int bin = 1; bin <= nBins; bin++) {
              int bin0 = std::max(bin - integ, 0);
              int bin1 = std::min(bin + integ, nBins);
              std::unique_ptr<TH1D> proj{src->ProjectionX("proj", bin0, bin1)};
              proj->ClearUnderflowAndOverflow();
              if (proj->GetEntries()) {
                unsigned int rebin = 1;
                while (proj->GetMaximum() < 50 && rebin < sizeof(RES_AXIS_BINS) / sizeof(RES_AXIS_BINS[0])) {
                  proj->Rebin(RES_AXIS_BINS[rebin - 1] / RES_AXIS_BINS[rebin]);
                  rebin++;
                }

                if (proj->GetEntries() < 20 || proj->GetRMS() < 0.00001) {
                  dst[0]->SetBinContent(bin, proj->GetRMS());
                  dst[0]->SetBinError(bin, std::sqrt(proj->GetRMS()));
                  dst[1]->SetBinContent(bin, proj->GetMean());
                  dst[1]->SetBinError(bin, std::sqrt(proj->GetRMS()));
                } else {
                  proj->GetXaxis()->SetRangeUser(proj->GetMean() - 6. * proj->GetRMS(), proj->GetMean() + 6. * proj->GetRMS());
                  proj->GetXaxis()->SetRangeUser(proj->GetMean() - 3. * proj->GetRMS(), proj->GetMean() + 3. * proj->GetRMS());
                  bool forceLogLike = proj->GetMaximum() < 20;
                  for (int k = forceLogLike ? 2 : 0; k < 3; k++) {
                    proj->Fit("gaus", forceLogLike || k == 2 ? "sQl" : k ? "sQww" : "sQ");
                    TF1* fitFunc = proj->GetFunction("gaus");

                    if (k && !forceLogLike) {
                      customGaus->SetParameters(fitFunc->GetParameter(0), fitFunc->GetParameter(1), fitFunc->GetParameter(2));
                      proj->Fit(customGaus.get(), "sQ");
                      fitFunc = customGaus.get();
                    }

                    const float sigma = fabs(fitFunc->GetParameter(2));
                    dst[0]->SetBinContent(bin, sigma);
                    dst[1]->SetBinContent(bin, fitFunc->GetParameter(1));
                    dst[0]->SetBinError(bin, fitFunc->GetParError(2));
                    dst[1]->SetBinError(bin, fitFunc->GetParError(1));

                    const bool fail1 = sigma <= 0.f;
                    const bool fail2 = fabs(proj->GetMean() - dst[1]->GetBinContent(bin)) > std::min<float>(p ? PULL_AXIS : mConfig.nativeFitResolutions ? RES_AXES_NATIVE[j] : RES_AXES[j], 3.f * proj->GetRMS());
                    const bool fail3 = dst[0]->GetBinContent(bin) > 3.f * proj->GetRMS() || dst[0]->GetBinError(bin) > 1 || dst[1]->GetBinError(bin) > 1;
                    const bool fail4 = fitFunc->GetParameter(0) < proj->GetMaximum() / 5.;
                    const bool fail = fail1 || fail2 || fail3 || fail4;
                    // if (p == 0 && ii == 4 && j == 2) DrawHisto(proj, Form("Hist_bin_%d-%d_vs_%d____%d_%d___%f-%f___%f-%f___%d.pdf", p, j, ii, bin, k, dst[0]->GetBinContent(bin), proj->GetRMS(), dst[1]->GetBinContent(bin), proj->GetMean(), (int) fail), "");

                    if (!fail) {
                      break;
                    } else if (k >= 2) {
                      dst[0]->SetBinContent(bin, proj->GetRMS());
                      dst[0]->SetBinError(bin, std::sqrt(proj->GetRMS()));
                      dst[1]->SetBinContent(bin, proj->GetMean());
                      dst[1]->SetBinError(bin, std::sqrt(proj->GetRMS()));
                    }
                  }
                }
              } else {
                dst[0]->SetBinContent(bin, 0.f);
                dst[0]->SetBinError(bin, 0.f);
                dst[1]->SetBinContent(bin, 0.f);
                dst[1]->SetBinError(bin, 0.f);
              }
            }
            if (ii == 0) {
              dstIntegral = src->ProjectionX(mConfig.nativeFitResolutions ? PARAMETER_NAMES_NATIVE[j] : PARAMETER_NAMES[j], 0, nBins + 1);
              unsigned int rebin = 1;
              while (dstIntegral->GetMaximum() < 50 && rebin < sizeof(RES_AXIS_BINS) / sizeof(RES_AXIS_BINS[0])) {
                dstIntegral->Rebin(RES_AXIS_BINS[rebin - 1] / RES_AXIS_BINS[rebin]);
                rebin++;
              }
            }
          }
          if (ii == 0) {
            if (mConfig.inputHistogramsOnly) {
              dstIntegral = createGarbageCollected<TH1D>();
            }
            sprintf(fname, p ? "IntPull%s" : "IntRes%s", VSPARAMETER_NAMES[j]);
            sprintf(name, p ? "%s Pull" : "%s Resolution", p || mConfig.nativeFitResolutions ? PARAMETER_NAMES_NATIVE[j] : PARAMETER_NAMES[j]);
            dstIntegral->SetName(fname);
            dstIntegral->SetTitle(name);
          }
          if (mConfig.enableLocalOutput || mConfig.shipToQCAsCanvas) {
            pad->cd();
          }
          int numColor = 0;
          float tmpMax = -1000.;
          float tmpMin = 1000.;

          for (int l = 0; l < 2; l++) {
            for (int k = 0; k < ConfigNumInputs; k++) {
              TH1F* e = dst[l];
              if (GetHist(e, tin, k, nNewInput) == nullptr) {
                continue;
              }
              if (nNewInput && k == 0 && ii != 5) {
                if (p == 0) {
                  e->Scale(mConfig.nativeFitResolutions ? SCALE_NATIVE[j] : SCALE[j]);
                }
              }
              if (ii == 4) {
                e->GetXaxis()->SetRangeUser(0.2, PT_MAX);
              } else if (LOG_PT_MIN > 0 && ii == 5) {
                e->GetXaxis()->SetRangeUser(LOG_PT_MIN, PT_MAX);
              } else if (ii == 5) {
                e->GetXaxis()->SetRange(1, 0);
              }
              e->SetMinimum(-1111);
              e->SetMaximum(-1111);

              if (e->GetMaximum() > tmpMax) {
                tmpMax = e->GetMaximum();
              }
              if (e->GetMinimum() < tmpMin) {
                tmpMin = e->GetMinimum();
              }
            }
          }

          float tmpSpan;
          tmpSpan = tmpMax - tmpMin;
          tmpMax += tmpSpan * .02;
          tmpMin -= tmpSpan * .02;
          if (j == 2 && i < 3) {
            tmpMax += tmpSpan * 0.13 * ConfigNumInputs;
          }

          for (int k = 0; k < ConfigNumInputs; k++) {
            for (int l = 0; l < 2; l++) {
              TH1F* e = dst[l];
              if (!mConfig.inputHistogramsOnly && k == 0) {
                sprintf(name, p ? "%s Pull" : "%s Resolution", p || mConfig.nativeFitResolutions ? PARAMETER_NAMES_NATIVE[j] : PARAMETER_NAMES[j]);
                e->SetTitle(name);
                e->SetStats(kFALSE);
                if (tout) {
                  if (l == 0) {
                    mRes2[j][i]->SetOption("colz");
                    mRes2[j][i]->Write();
                  }
                  e->Write();
                }
              } else if (GetHist(e, tin, k, nNewInput) == nullptr) {
                continue;
              }
              e->SetMaximum(tmpMax);
              e->SetMinimum(tmpMin);
              e->SetLineWidth(1);
              e->SetLineStyle(CONFIG_DASHED_MARKERS ? k + 1 : 1);
              SetAxisSize(e);
              e->GetYaxis()->SetTitle(p ? AXIS_TITLES_PULL[j] : mConfig.nativeFitResolutions ? AXIS_TITLES_NATIVE[j] : AXIS_TITLES[j]);
              e->GetXaxis()->SetTitle(XAXIS_TITLES[i]);
              if (LOG_PT_MIN > 0 && ii == 5) {
                e->GetXaxis()->SetRangeUser(LOG_PT_MIN, PT_MAX);
              }

              if (j == 0) {
                e->GetYaxis()->SetTitleOffset(1.5);
              } else if (j < 3) {
                e->GetYaxis()->SetTitleOffset(1.4);
              }
              if (qcout && !mConfig.shipToQCAsCanvas) {
                qcout->Add(e);
              }
              if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
                continue;
              }

              e->SetMarkerColor(kBlack);
              e->SetLineColor(colorNums[numColor++ % COLORCOUNT]);
              e->Draw(k || l ? "same" : "");
              if (j == 0) {
                GetName(fname, k);
                if (p) {
                  sprintf(name, "%s%s", fname, l ? "Mean" : "Pull");
                } else {
                  sprintf(name, "%s%s", fname, l ? "Mean" : "Resolution");
                }
                leg->AddEntry(e, name, "l");
              }
            }
          }
          if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
            continue;
          }

          if (ii == 5) {
            pad->SetLogx();
          }
          can->cd();
          if (j == 4) {
            ChangePadTitleSize(pad, 0.056);
          }
        }
        if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
          continue;
        }

        leg->Draw();

        if (qcout) {
          qcout->Add(can);
        }
        if (!mConfig.enableLocalOutput) {
          continue;
        }
        doPerfFigure(0.2, 0.295, 0.025);
        can->Print(Form(p ? "plots/pull_vs_%s.pdf" : "plots/res_vs_%s.pdf", VSPARAMETER_NAMES[ii]));
        if (mConfig.writeRootFiles) {
          can->Print(Form(p ? "plots/pull_vs_%s.root" : "plots/res_vs_%s.root", VSPARAMETER_NAMES[ii]));
        }
      }
    }

    // Process Integral Resolution Histogreams
    for (int p = 0; p < 2; p++) {
      if ((p == 0 && (mQATasks & taskTrackingRes) == 0) || (p == 1 && (mQATasks & taskTrackingResPull) == 0)) {
        continue;
      }
      TCanvas* can = p ? mCPull[6] : mCRes[6];
      for (int i = 0; i < 5; i++) {
        TPad* pad = p ? mPPull[6][i] : mPRes[6][i];
        TH1D* hist = p ? pullIntegral[i] : resIntegral[i];
        int numColor = 0;
        if (mConfig.enableLocalOutput || mConfig.shipToQCAsCanvas) {
          pad->cd();
        }
        if (!mConfig.inputHistogramsOnly && mcAvail) {
          TH1D* e = hist;
          if (e && e->GetEntries()) {
            e->Fit("gaus", "sQ");
          }
        }

        float tmpMax = 0;
        for (int k = 0; k < ConfigNumInputs; k++) {
          TH1D* e = hist;
          if (GetHist(e, tin, k, nNewInput) == nullptr) {
            continue;
          }
          e->SetMaximum(-1111);
          if (e->GetMaximum() > tmpMax) {
            tmpMax = e->GetMaximum();
          }
        }

        for (int k = 0; k < ConfigNumInputs; k++) {
          TH1D* e = hist;
          if (GetHist(e, tin, k, nNewInput) == nullptr) {
            continue;
          }
          e->SetMaximum(tmpMax * 1.02);
          e->SetMinimum(tmpMax * -0.02);
          if (tout && !mConfig.inputHistogramsOnly && k == 0) {
            e->Write();
          }
          if (qcout && !mConfig.shipToQCAsCanvas) {
            qcout->Add(e);
          }
          if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
            continue;
          }

          e->SetLineColor(colorNums[numColor++ % COLORCOUNT]);
          e->Draw(k == 0 ? "" : "same");
        }
        if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
          continue;
        }
        can->cd();
      }
      if (qcout) {
        qcout->Add(can);
      }
      if (!mConfig.enableLocalOutput) {
        continue;
      }

      can->Print(p ? "plots/pull_integral.pdf" : "plots/res_integral.pdf");
      if (mConfig.writeRootFiles) {
        can->Print(p ? "plots/pull_integral.root" : "plots/res_integral.root");
      }
    }
  }

  unsigned long long int attachClusterCounts[N_CLS_HIST];
  if (mQATasks & taskClusterAttach) {
    // Process Cluster Attachment Histograms
    if (mConfig.inputHistogramsOnly == 0) {
      for (int i = N_CLS_HIST; i < N_CLS_TYPE * N_CLS_HIST - 1; i++) {
        mClusters[i]->Sumw2(true);
      }
      double totalVal = 0;
      if (!CLUST_HIST_INT_SUM) {
        for (int j = 0; j < mClusters[N_CLS_HIST - 1]->GetXaxis()->GetNbins() + 2; j++) {
          totalVal += mClusters[N_CLS_HIST - 1]->GetBinContent(j);
        }
      }
      if (totalVal == 0.) {
        totalVal = 1.;
      }
      for (int i = 0; i < N_CLS_HIST; i++) {
        double val = 0;
        for (int j = 0; j < mClusters[i]->GetXaxis()->GetNbins() + 2; j++) {
          val += mClusters[i]->GetBinContent(j);
          mClusters[2 * N_CLS_HIST - 1 + i]->SetBinContent(j, val / totalVal);
        }
        attachClusterCounts[i] = val;
      }

      if (!CLUST_HIST_INT_SUM) {
        for (int i = 0; i < N_CLS_HIST; i++) {
          mClusters[2 * N_CLS_HIST - 1 + i]->SetMaximum(1.02);
          mClusters[2 * N_CLS_HIST - 1 + i]->SetMinimum(-0.02);
        }
      }

      for (int i = 0; i < N_CLS_HIST - 1; i++) {
        mClusters[N_CLS_HIST + i]->Divide(mClusters[i], mClusters[N_CLS_HIST - 1], 1, 1, "B");
        mClusters[N_CLS_HIST + i]->SetMinimum(-0.02);
        mClusters[N_CLS_HIST + i]->SetMaximum(1.02);
      }
    }

    float tmpMax[2] = {0, 0}, tmpMin[2] = {0, 0};
    for (int l = 0; l <= CLUST_HIST_INT_SUM; l++) {
      for (int k = 0; k < ConfigNumInputs; k++) {
        TH1* e = mClusters[l ? (N_CLS_TYPE * N_CLS_HIST - 2) : (N_CLS_HIST - 1)];
        if (GetHist(e, tin, k, nNewInput) == nullptr) {
          continue;
        }
        e->SetMinimum(-1111);
        e->SetMaximum(-1111);
        if (l == 0) {
          e->GetXaxis()->SetRange(2, AXIS_BINS[4]);
        }
        if (e->GetMaximum() > tmpMax[l]) {
          tmpMax[l] = e->GetMaximum();
        }
        if (e->GetMinimum() < tmpMin[l]) {
          tmpMin[l] = e->GetMinimum();
        }
      }
      for (int k = 0; k < ConfigNumInputs; k++) {
        for (int i = 0; i < N_CLS_HIST; i++) {
          TH1* e = mClusters[l ? (2 * N_CLS_HIST - 1 + i) : i];
          if (GetHist(e, tin, k, nNewInput) == nullptr) {
            continue;
          }
          e->SetMaximum(tmpMax[l] * 1.02);
          e->SetMinimum(tmpMax[l] * -0.02);
        }
      }
    }

    for (int i = 0; i < N_CLS_TYPE; i++) {
      if (mConfig.enableLocalOutput || mConfig.shipToQCAsCanvas) {
        mPClust[i]->cd();
        mPClust[i]->SetLogx();
      }
      int begin = i == 2 ? (2 * N_CLS_HIST - 1) : i == 1 ? N_CLS_HIST : 0;
      int end = i == 2 ? (3 * N_CLS_HIST - 1) : i == 1 ? (2 * N_CLS_HIST - 1) : N_CLS_HIST;
      int numColor = 0;
      for (int k = 0; k < ConfigNumInputs; k++) {
        for (int j = end - 1; j >= begin; j--) {
          TH1* e = mClusters[j];
          if (GetHist(e, tin, k, nNewInput) == nullptr) {
            continue;
          }

          e->SetTitle(CLUSTER_TITLES[i]);
          e->GetYaxis()->SetTitle(i == 0 ? "Number of TPC clusters" : i == 1 ? "Fraction of TPC clusters" : CLUST_HIST_INT_SUM ? "Total TPC clusters (integrated)" : "Fraction of TPC clusters (integrated)");
          e->GetXaxis()->SetTitle("#it{p}_{Tmc} (GeV/#it{c})");
          e->GetXaxis()->SetTitleOffset(1.1);
          e->GetXaxis()->SetLabelOffset(-0.005);
          if (tout && !mConfig.inputHistogramsOnly && k == 0) {
            e->Write();
          }
          e->SetStats(kFALSE);
          e->SetLineWidth(1);
          e->SetLineStyle(CONFIG_DASHED_MARKERS ? j + 1 : 1);
          if (i == 0) {
            e->GetXaxis()->SetRange(2, AXIS_BINS[4]);
          }
          if (qcout && !mConfig.shipToQCAsCanvas) {
            qcout->Add(e);
          }
          if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
            continue;
          }

          e->SetMarkerColor(kBlack);
          e->SetLineColor(colorNums[numColor++ % COLORCOUNT]);
          e->Draw(j == end - 1 && k == 0 ? "" : "same");
          GetName(fname, k);
          sprintf(name, "%s%s", fname, CLUSTER_NAMES[j - begin]);
          mLClust[i]->AddEntry(e, name, "l");
        }
      }
      if (ConfigNumInputs == 1) {
        TH1* e = reinterpret_cast<TH1F*>(mClusters[begin + CL_att_adj]->Clone());
        e->Add(mClusters[begin + CL_prot], -1);
        if (qcout && !mConfig.shipToQCAsCanvas) {
          qcout->Add(e);
        }
        if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
          continue;
        }

        e->SetLineColor(colorNums[numColor++ % COLORCOUNT]);
        e->Draw("same");
        mLClust[i]->AddEntry(e, "Removed (Strategy A)", "l");
      }
      if (!mConfig.enableLocalOutput && !mConfig.shipToQCAsCanvas) {
        continue;
      }

      mLClust[i]->Draw();

      if (qcout) {
        qcout->Add(mCClust[i]);
      }
      if (!mConfig.enableLocalOutput) {
        continue;
      }
      doPerfFigure(i != 2 ? 0.37 : 0.6, 0.295, 0.030);
      mCClust[i]->cd();
      mCClust[i]->Print(i == 2 ? "plots/clusters_integral.pdf" : i == 1 ? "plots/clusters_relative.pdf" : "plots/clusters.pdf");
      if (mConfig.writeRootFiles) {
        mCClust[i]->Print(i == 2 ? "plots/clusters_integral.root" : i == 1 ? "plots/clusters_relative.root" : "plots/clusters.root");
      }
    }
  }

  // Process cluster count statistics
  if ((mQATasks & taskClusterCounts) && !mHaveExternalHists && !mConfig.clusterRejectionHistograms && !mConfig.inputHistogramsOnly) {
    DoClusterCounts(attachClusterCounts);
  }
  if ((qcout || tout) && (mQATasks & taskClusterCounts) && mConfig.clusterRejectionHistograms) {
    for (unsigned int i = 0; i < mHistClusterCount.size(); i++) {
      if (tout) {
        mHistClusterCount[i]->Write();
      }
      if (qcout) {
        qcout->Add(mHistClusterCount[i]);
      }
    }
  }

  if (mQATasks & taskTrackStatistics) {
    // Process track statistic histograms
    float tmpMax = 0.;
    for (int k = 0; k < ConfigNumInputs; k++) {
      TH1F* e = mTracks;
      if (GetHist(e, tin, k, nNewInput) == nullptr) {
        continue;
      }
      e->SetMaximum(-1111);
      if (e->GetMaximum() > tmpMax) {
        tmpMax = e->GetMaximum();
      }
    }
    mPTracks->cd();
    mPTracks->SetLogx();
    for (int k = 0; k < ConfigNumInputs; k++) {
      TH1F* e = mTracks;
      if (GetHist(e, tin, k, nNewInput) == nullptr) {
        continue;
      }
      if (tout && !mConfig.inputHistogramsOnly && k == 0) {
        e->Write();
      }
      e->SetMaximum(tmpMax * 1.02);
      e->SetMinimum(tmpMax * -0.02);
      e->SetStats(kFALSE);
      e->SetLineWidth(1);
      e->GetYaxis()->SetTitle("a.u.");
      e->GetXaxis()->SetTitle("#it{p}_{Tmc} (GeV/#it{c})");
      if (qcout) {
        qcout->Add(e);
      }
      e->SetMarkerColor(kBlack);
      e->SetLineColor(colorNums[k % COLORCOUNT]);
      e->Draw(k == 0 ? "" : "same");
      GetName(fname, k);
      sprintf(name, "%sTrack Pt", fname);
      mLTracks->AddEntry(e, name, "l");
    }
    mLTracks->Draw();
    mCTracks->cd();
    mCTracks->Print("plots/tracks.pdf");
    if (mConfig.writeRootFiles) {
      mCTracks->Print("plots/tracks.root");
    }
    tmpMax = 0.;
    for (int k = 0; k < ConfigNumInputs; k++) {
      TH1F* e = mNCl;
      if (GetHist(e, tin, k, nNewInput) == nullptr) {
        continue;
      }
      e->SetMaximum(-1111);
      if (e->GetMaximum() > tmpMax) {
        tmpMax = e->GetMaximum();
      }
    }
    mPNCl->cd();
    for (int k = 0; k < ConfigNumInputs; k++) {
      TH1F* e = mNCl;
      if (GetHist(e, tin, k, nNewInput) == nullptr) {
        continue;
      }
      if (tout && !mConfig.inputHistogramsOnly && k == 0) {
        e->Write();
      }
      e->SetMaximum(tmpMax * 1.02);
      e->SetMinimum(tmpMax * -0.02);
      e->SetStats(kFALSE);
      e->SetLineWidth(1);
      e->GetYaxis()->SetTitle("a.u.");
      e->GetXaxis()->SetTitle("NClusters");
      if (qcout) {
        qcout->Add(e);
      }
      e->SetMarkerColor(kBlack);
      e->SetLineColor(colorNums[k % COLORCOUNT]);
      e->Draw(k == 0 ? "" : "same");
      GetName(fname, k);
      sprintf(name, "%sNClusters", fname);
      mLNCl->AddEntry(e, name, "l");
    }
    mLNCl->Draw();
    mCNCl->cd();
    mCNCl->Print("plots/nClusters.pdf");
    if (mConfig.writeRootFiles) {
      mCNCl->Print("plots/nClusters.root");
    }
  }

  if (tout && !mConfig.inputHistogramsOnly && mConfig.writeMCLabels) {
    gInterpreter->GenerateDictionary("vector<vector<int>>", "");
    tout->WriteObject(&mcEffBuffer, "mcEffBuffer");
    tout->WriteObject(&mcLabelBuffer, "mcLabelBuffer");
    remove("AutoDict_vector_vector_int__.cxx");
    remove("AutoDict_vector_vector_int___cxx_ACLiC_dict_rdict.pcm");
    remove("AutoDict_vector_vector_int___cxx.d");
    remove("AutoDict_vector_vector_int___cxx.so");
  }

  if (tout) {
    tout->Close();
  }
  for (unsigned int i = 0; i < mConfig.compareInputs.size(); i++) {
    tin[i]->Close();
  }
  if (!qcout) {
    clearGarbagageCollector();
  }
  return (0);
}

void GPUQA::PrintClusterCount(int mode, int& num, const char* name, unsigned long long int n, unsigned long long int normalization)
{
  if (mode == 2) {
    // do nothing, just count num
  } else if (mode == 1) {
    char name2[128];
    sprintf(name2, "clusterCount%d", num);
    createHist(mHistClusterCount[num], name2, name, 1000, 0, mConfig.histMaxNClusters, 1000, 0, 100);
  } else if (mode == 0) {
    if (normalization && mConfig.enableLocalOutput) {
      printf("\t%35s: %'12llu (%6.2f%%)\n", name, n, 100.f * n / normalization);
    }
    if (mConfig.clusterRejectionHistograms) {
      float ratio = 100.f * n / std::max(normalization, 1llu);
      mHistClusterCount[num]->Fill(n, ratio, 1);
    }
  }
  num++;
}

int GPUQA::DoClusterCounts(unsigned long long int* attachClusterCounts, int mode)
{
  int num = 0;
  if (mcPresent() && (mQATasks & taskClusterAttach) && attachClusterCounts) {
    for (int i = 0; i < N_CLS_HIST; i++) {
      PrintClusterCount(mode, num, CLUSTER_NAMES[i], attachClusterCounts[i], mClusterCounts.nTotal);
    }
    PrintClusterCount(mode, num, "Unattached", attachClusterCounts[N_CLS_HIST - 1] - attachClusterCounts[CL_att_adj], mClusterCounts.nTotal);
    PrintClusterCount(mode, num, "Removed (Strategy A)", attachClusterCounts[CL_att_adj] - attachClusterCounts[CL_prot], mClusterCounts.nTotal); // Attached + Adjacent (also fake) - protected
    PrintClusterCount(mode, num, "Unaccessible", mClusterCounts.nUnaccessible, mClusterCounts.nTotal);                                           // No contribution from track >= 10 MeV, unattached or fake-attached/adjacent
  } else {
    PrintClusterCount(mode, num, "All Clusters", mClusterCounts.nTotal, mClusterCounts.nTotal);
    PrintClusterCount(mode, num, "Used in Physics", mClusterCounts.nPhysics, mClusterCounts.nTotal);
    PrintClusterCount(mode, num, "Protected", mClusterCounts.nProt, mClusterCounts.nTotal);
    PrintClusterCount(mode, num, "Unattached", mClusterCounts.nUnattached, mClusterCounts.nTotal);
    PrintClusterCount(mode, num, "Removed (Strategy A)", mClusterCounts.nTotal - mClusterCounts.nUnattached - mClusterCounts.nProt, mClusterCounts.nTotal);
  }

  PrintClusterCount(mode, num, "Merged Loopers (Afterburner)", mClusterCounts.nMergedLooper, mClusterCounts.nTotal);
  PrintClusterCount(mode, num, "High Inclination Angle", mClusterCounts.nHighIncl, mClusterCounts.nTotal);
  PrintClusterCount(mode, num, "Rejected", mClusterCounts.nRejected, mClusterCounts.nTotal);
  PrintClusterCount(mode, num, "Tube (> 200 MeV)", mClusterCounts.nTube, mClusterCounts.nTotal);
  PrintClusterCount(mode, num, "Tube (< 200 MeV)", mClusterCounts.nTube200, mClusterCounts.nTotal);
  PrintClusterCount(mode, num, "Looping Legs", mClusterCounts.nLoopers, mClusterCounts.nTotal);
  PrintClusterCount(mode, num, "Low Pt < 50 MeV", mClusterCounts.nLowPt, mClusterCounts.nTotal);
  PrintClusterCount(mode, num, "Low Pt < 200 MeV", mClusterCounts.n200MeV, mClusterCounts.nTotal);

  if (mcPresent() && (mQATasks & taskClusterAttach)) {
    PrintClusterCount(mode, num, "Tracks > 400 MeV", mClusterCounts.nAbove400, mClusterCounts.nTotal);
    PrintClusterCount(mode, num, "Fake Removed (> 400 MeV)", mClusterCounts.nFakeRemove400, mClusterCounts.nAbove400);
    PrintClusterCount(mode, num, "Full Fake Removed (> 400 MeV)", mClusterCounts.nFullFakeRemove400, mClusterCounts.nAbove400);
    PrintClusterCount(mode, num, "Tracks < 40 MeV", mClusterCounts.nBelow40, mClusterCounts.nTotal);
    PrintClusterCount(mode, num, "Fake Protect (< 40 MeV)", mClusterCounts.nFakeProtect40, mClusterCounts.nBelow40);
  }
  return num;
}
