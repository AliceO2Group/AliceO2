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

/// \file GPUDisplay.h
/// \author David Rohr

#ifndef GPUDISPLAY_H
#define GPUDISPLAY_H

#include "frontend/GPUDisplayFrontend.h"
#include "backend/GPUDisplayBackend.h"
#include "GPUDisplayInterface.h"
#include "GPUSettings.h"
#include "DataFormatsTPC/Constants.h"

#include <atomic>
#include <array>
#include <tuple>
#include <mutex>
#include <condition_variable>

union hmm_mat4; // From HandMadeMath

#include "utils/timer.h"
#include "utils/vecpod.h"
#include "utils/qsem.h"

namespace o2::gpu
{
class GPUTPCTracker;
struct GPUParam;
class GPUQA;
class GPUTRDGeometry;
class GPUTPCGMPropagator;

class GPUDisplay : public GPUDisplayInterface
{
  friend class GPUDisplayFrontend;

 public:
  GPUDisplay(GPUDisplayFrontend* frontend, GPUChainTracking* chain, GPUQA* qa, const GPUParam* param = nullptr, const GPUCalibObjectsConst* calib = nullptr, const GPUSettingsDisplay* config = nullptr, const GPUSettingsProcessing* proc = nullptr);
  GPUDisplay(const GPUDisplay&) = delete;
  ~GPUDisplay() override = default;

  int32_t StartDisplay() override;
  void ExitDisplay();
  void ShowNextEvent(const GPUTrackingInOutPointers* ptrs = nullptr) override;
  void BlockTillNextEvent() override;
  void UnblockDisplay();
  void WaitTillEventShown() override;
  void SetCollisionFirstCluster(uint32_t collision, int32_t sector, int32_t cluster) override;
  void UpdateCalib(const GPUCalibObjectsConst* calib) override { mCalib = calib; }
  void UpdateParam(const GPUParam* param) override { mParam = param; }

  void HandleKey(uint8_t key);
  int32_t DrawGLScene(bool suspended = false);
  void HandleExternalGUIKey(int32_t key);
  int32_t InitDisplay(bool initFailure = false);
  void ResizeScene(int32_t width, int32_t height, bool init = false);
  void checkAndApplyConfigChanges(); // Set update flags / apply the settings that changed compared to mOldCfg

  const GPUSettingsDisplayRenderer& cfgR() const { return mCfgR; }
  const GPUSettingsDisplayLight& cfgL() const { return mCfgL; }
  const GPUSettingsDisplayHeavy& cfgH() const { return mCfgH; }
  const GPUSettingsDisplay& cfg() const { return mConfig; }
  bool useMultiVBO() const { return mUseMultiVBO; }
  int32_t updateDrawCommands() const { return mUpdateDrawCommands; }
  int32_t updateRenderPipeline() const { return mUpdateRenderPipeline; }
  GPUDisplayBackend* backend() const { return mBackend.get(); }
  vecpod<int32_t>* vertexBufferStart() { return mVertexBufferStart; }
  const vecpod<uint32_t>* vertexBufferCount() const { return mVertexBufferCount; }
  struct vtx {
    float x, y, z;
    vtx(float a, float b, float c) : x(a), y(b), z(c) {}
  };
  vecpod<vtx>* vertexBuffer() { return mVertexBuffer; }
  const GPUParam* param() { return mParam; }
  const GPUSettingsProcessing& GetProcessingSettings() const { return mProcessingSettings; }
  GPUDisplayFrontend* frontend() { return mFrontend; }
  int32_t& drawTextFontSize() { return mDrawTextFontSize; }

  struct control {
    std::atomic<int32_t> mDisplayControl = 0; // Control for next event (=1) or quit (=2)
    std::atomic<bool> mResetScene = false;
    int32_t mPrintInfoText = 1;
    bool mPrintInfoTextAlways = false;
    bool mRequestScreenshot = false;
    int32_t mAnimateScreenshot = 0;
    bool mAnimationChangeConfig = true;
    bool mGuiPrintInfo = false;  // Show the info of options changed with Qt GUI elements (not for key presses in the GUI)
    bool mRotateModel = false;   // Rotate the model instead of the camera, like holding RCTRL (set in the Qt GUI)
    bool mRotateAroundZ = false; // Rotate the TPC around the beamline, like holding RALT (set in the Qt GUI)
    float mGuiScale = 1.f;       // Scale factor of text and elements of the Qt GUI (on top of the system scaling), Ctrl + mouse wheel / Ctrl + 0 in the GUI
    bool dummy = false;          // dummy value to reference for keys that do not change variables
  } mCfgC;

 private:
  void buildConfig();
  void buildConfigKeys();
  void buildConfigInfo();
  void buildConfigQtGUI();

  static constexpr int32_t NSECTORS = o2::tpc::constants::MAXSECTOR;
  static constexpr float GL_SCALE_FACTOR = (1.f / 100.f);

  static constexpr const int32_t N_POINTS_TYPE = 15;
  static constexpr const int32_t N_POINTS_TYPE_TPC = 9;
  static constexpr const int32_t N_POINTS_TYPE_TRD = 2;
  static constexpr const int32_t N_POINTS_TYPE_TOF = 2;
  static constexpr const int32_t N_POINTS_TYPE_ITS = 2;
  static constexpr const int32_t N_LINES_TYPE = 7;
  static constexpr const int32_t N_FINAL_TYPE = 4;
  static constexpr int32_t TRACK_TYPE_ID_LIMIT = 100;
  enum PointTypes { tCLUSTER = 0,
                    tINITLINK = 1,
                    tLINK = 2,
                    tSEED = 3,
                    tTRACKLET = 4,
                    tSECTORTRACK = 5,
                    tEXTRAPOLATEDTRACK = 6,
                    tFINALTRACK = 7,
                    tMARKED = 8,
                    tTRDCLUSTER = 9,
                    tTRDATTACHED = 10,
                    tTOFCLUSTER = 11,
                    tTOFATTACHED = 12,
                    tITSCLUSTER = 13,
                    tITSATTACHED = 14 };
  enum LineTypes { RESERVED = 0 /*1 -- 6 = INITLINK to GLOBALTRACK*/ };

  using vboList = GPUDisplayBackend::vboList;

  struct threadVertexBuffer {
    vecpod<vtx> buffer;
    vecpod<int32_t> start[N_FINAL_TYPE];
    vecpod<uint32_t> count[N_FINAL_TYPE];
    std::pair<vecpod<int32_t>*, vecpod<uint32_t>*> vBuf[N_FINAL_TYPE];
    threadVertexBuffer() : buffer()
    {
      for (int32_t i = 0; i < N_FINAL_TYPE; i++) {
        vBuf[i].first = start + i;
        vBuf[i].second = count + i;
      }
    }
    void clear()
    {
      for (int32_t i = 0; i < N_FINAL_TYPE; i++) {
        start[i].clear();
        count[i].clear();
      }
    }
  };

  class opengl_spline
  {
   public:
    opengl_spline() : ma(), mb(), mc(), md(), mx() {}
    void create(const vecpod<float>& x, const vecpod<float>& y);
    float evaluate(float x);
    void setVerbose() { mVerbose = true; }

   private:
    vecpod<float> ma, mb, mc, md, mx;
    bool mVerbose = false;
  };

  static const GPUSettingsDisplay& GetConfig(GPUChainTracking* chain);
  static const GPUSettingsProcessing& GetProcessingConfig(GPUChainTracking* chain);

  void DrawGLScene_internal(bool suspended, float animateTime = -1.f, bool renderToMixBuffer = false); // suspended: return before drawing
  void DrawGLScene_updateEventData();
  void DrawGLScene_cameraAndAnimation(bool suspended, float animateTime, float& mixSlaveImage, hmm_mat4& nextViewMatrix);
  size_t DrawGLScene_updateVertexList();
  void DrawGLScene_drawCommands();
  int32_t InitDisplay_internal();
  int32_t getNumThreads();
  void updateOptions();
  void disableUnsupportedOptions();
  int32_t buildTrackFilter();
  const GPUTPCTracker& sectorTracker(int32_t iSector);
  const GPUTRDGeometry* trdGeometry();
  const GPUTrackingInOutPointers* mIOPtrs = nullptr;
  void insertVertexList(std::pair<vecpod<int32_t>*, vecpod<uint32_t>*>& vBuf, size_t first, size_t last);
  void insertVertexList(int32_t iSector, size_t first, size_t last);
  template <typename... Args>
  void SetInfo(Args... args);
  void PrintGLHelpText(float colorValue);
  void calcXYZ(const float*);
  void mAnimationCloseAngle(float& newangle, float lastAngle);
  void mAnimateCloseQuaternion(float* v, float lastx, float lasty, float lastz, float lastw);
  void setAnimationPoint();
  void resetAnimation();
  void removeAnimationPoint();
  void startAnimation();
  int32_t animateCamera(bool suspended, float& animateTime, float& mixSlaveImage, hmm_mat4& nextViewMatrix);
  void showInfo(const char* info);
  void ActivateColor();
  void SetColorTRD();
  void SetColorTOF();
  void SetColorITS();
  void SetColorClusters();
  void SetColorInitLinks();
  void SetColorLinks();
  void SetColorSeeds();
  void SetColorTracklets();
  void SetColorTracks();
  void SetColorExtrapolatedTracks();
  void SetColorFinal();
  void SetColorGrid();
  void SetColorGridTRD();
  void SetColorMarked();
  void SetCollisionColor(int32_t col);
  void updateConfig();
  void drawPointLinestrip(int32_t iSector, int32_t cid, int32_t id, int32_t id_limit = TRACK_TYPE_ID_LIMIT);
  void DrawClusters(int32_t iSector);
  vboList DrawSpacePointsTRD(int32_t iSector, int32_t select, int32_t iCol);
  vboList DrawSpacePointsTOF(int32_t iSector, int32_t select, int32_t iCol);
  vboList DrawSpacePointsITS(int32_t iSector, int32_t select, int32_t iCol);
  vboList DrawLinks(const GPUTPCTracker& tracker, int32_t id, bool dodown = false);
  vboList DrawSeeds(const GPUTPCTracker& tracker);
  vboList DrawTracklets(const GPUTPCTracker& tracker);
  vboList DrawTracks(const GPUTPCTracker& tracker, int32_t global);
  void DrawTrackITS(int32_t trackId, int32_t iSector);
  GPUDisplay::vboList DrawFinalITS();
  template <class T>
  void DrawFinal(int32_t iSector, int32_t /*iCol*/, const GPUTPCGMPropagator* prop, std::array<vecpod<int32_t>, 2>& trackList, threadVertexBuffer& threadBuffer);
  vboList DrawGrid(const GPUTPCTracker& tracker);
  vboList DrawGridTRD(int32_t sector);
  void DoScreenshot(const char* filename, std::vector<char>& pixels, float animateTime = -1.f);
  void PrintHelp();
  void createQuaternionFromMatrix(float* v, const float* mat);
  void drawVertices(const vboList& v, const GPUDisplayBackend::drawType t);
  void GLPrintText(const char* s, float x, float y, float r, float g, float b, float a, uint32_t nCharacters = 0, bool fromBotton = true); // nCharacters > 0: print at most that many characters

  GPUDisplayFrontend* mFrontend = nullptr;
  std::unique_ptr<GPUDisplayBackend> mBackend;
  GPUChainTracking* mChain = nullptr;
  const GPUParam* mParam = nullptr;
  const GPUCalibObjectsConst* mCalib = nullptr;
  const GPUSettingsDisplay& mConfig;
  GPUSettingsDisplayLight mCfgL;
  GPUSettingsDisplayHeavy mCfgH;
  GPUSettingsDisplayRenderer mCfgR;
  std::tuple<GPUSettingsDisplayHeavy, GPUSettingsDisplayLight, GPUSettingsDisplayRenderer> mOldCfg; // Settings after the last draw, for checkAndApplyConfigChanges
  const GPUSettingsProcessing& mProcessingSettings;
  GPUQA* mQA;

  qSem mSemLockDisplay;
  std::mutex mMutexLoadAndShowEvent;
  std::condition_variable mCVLoadAndShowEvent;

  int32_t mDrawTextFontSize = 0;

  int32_t mNDrawCalls = 0;

  bool mUseMultiVBO = false;

  std::array<float, 4> mDrawColor = {1.f, 1.f, 1.f, 1.f};

  float mAngleRollOrigin = -1e9;
  float mMaxClusterZ = -1;

  std::unique_ptr<hmm_mat4> mViewMatrix, mModelMatrix;
  float mXYZ[3];
  float mAngle[3];
  float mRPhiTheta[3];
  float mQuat[4];

  vecpod<std::array<int32_t, 37>> mOverlayTFClusters;
  int32_t mNCollissions = 1;

  vecpod<vtx> mVertexBuffer[NSECTORS];
  vecpod<int32_t> mVertexBufferStart[NSECTORS];
  vecpod<uint32_t> mVertexBufferCount[NSECTORS];
  std::vector<std::array<uint32_t, N_POINTS_TYPE_TPC>> mClusterBufferSizeCache[NSECTORS];

  std::unique_ptr<float4[]> mGlobalPosPtr;
  std::unique_ptr<float4[]> mGlobalPosPtrTRD;
  std::unique_ptr<float4[]> mGlobalPosPtrTRD2;
  std::unique_ptr<float4[]> mGlobalPosPtrITS;
  std::unique_ptr<float4[]> mGlobalPosPtrTOF;
  float4* mGlobalPos;
  float4* mGlobalPosTRD;
  float4* mGlobalPosTRD2;
  float4* mGlobalPosITS;
  float4* mGlobalPosTOF;
  int32_t mNMaxClusters = 0;
  int32_t mNMaxSpacePointsTRD = 0;
  int32_t mNMaxClustersITS = 0;
  int32_t mNMaxClustersTOF = 0;
  int32_t mCurrentClusters = 0;
  int32_t mCurrentSpacePointsTRD = 0;
  int32_t mCurrentClustersITS = 0;
  int32_t mCurrentClustersTOF = 0;
  vecpod<int32_t> mTRDTrackIds;
  vecpod<bool> mITSStandaloneTracks;
  std::vector<bool> mTrackFilter;

  volatile bool mUpdateTrackFilter = false;
  volatile bool mUpdateVertexLists = true;
  volatile bool mUpdateEventData = false;
  volatile bool mUpdateDrawCommands = true;
  volatile bool mUpdateRenderPipeline = false;
  volatile bool mLoadAndShowEvent = false;
  bool mTracksArePropagated = false;

  int32_t mAnimate = 0;
  HighResTimer mAnimationTimer;
  int32_t mAnimationFrame = 0;
  int32_t mAnimationLastBase = 0;
  int32_t mAnimationMixBase = -1; // Target keyframe of the current config blend (mixing), -1 = none
  int32_t mAnimationExport = 0;
  float mAnimationDelay = 2.f;
  vecpod<float> mAnimateVectors[9];
  vecpod<GPUSettingsDisplayLight> mAnimateConfig;
  opengl_spline mAnimationSplines[8];

  char mInfoText2[1024];
  HighResTimer mInfoText2Timer, mInfoHelpTimer;

  std::vector<threadVertexBuffer> mThreadBuffers;
  std::vector<std::vector<std::array<std::array<vecpod<int32_t>, 2>, NSECTORS>>> mThreadTracks;
  volatile int32_t mInitResult = 0;

  float mFPSScale = 1, mFPSScaleadjust = 0;
  int32_t mFramesDone = 0, mFramesDoneFPS = 0;
  HighResTimer mTimerFPS;
  vboList mGlDLLines[NSECTORS][N_LINES_TYPE];
  vecpod<std::array<vboList, N_FINAL_TYPE>> mGlDLFinal[NSECTORS];
  vboList mGlDLFinalITS;
  vecpod<vboList> mGlDLPoints[NSECTORS][N_POINTS_TYPE];
  vboList mGlDLGrid[NSECTORS];
  vboList mGlDLGridTRD[NSECTORS / 2];

  std::string mScreenshotFile;

  float mYFactor = 1.0f;
};
} // namespace o2::gpu

#endif
