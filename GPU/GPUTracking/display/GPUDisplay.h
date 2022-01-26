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

#include "GPUSettings.h"
#include "GPUDisplayFrontend.h"
#include "GPUDisplayBackend.h"

#ifndef GPUCA_BUILD_EVENT_DISPLAY

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplay
{
 public:
  GPUDisplay(void* frontend, void* chain, void* qa, const char* backend = "", const void* param = nullptr, const void* calib = nullptr, const void* config = nullptr) {}
  ~GPUDisplay() = default;
  GPUDisplay(const GPUDisplay&) = delete;

  int StartDisplay() { return 1; }
  void ShowNextEvent(const GPUTrackingInOutPointers* ptrs = nullptr) {}
  void WaitForNextEvent() {}
  void SetCollisionFirstCluster(unsigned int collision, int slice, int cluster) {}

  void HandleKey(unsigned char key) {}
  int DrawGLScene(bool mixAnimation = false, float mAnimateTime = -1.f) { return 1; }
  void HandleSendKey(int key) {}
  int InitDisplay(bool initFailure = false) { return 1; }
  void ExitDisplay() {}
  void ReSizeGLScene(int width, int height, bool init = false) {}

  const GPUDisplayBackend* backend() const { return nullptr; }
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else

#include "GPUChainTracking.h"
#include "../utils/vecpod.h"
#include "../utils/qsem.h"

#include <array>
#include "HandMadeMath.h"

#include "utils/timer.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCTracker;
struct GPUParam;
class GPUQA;

class GPUDisplay
{
 public:
  GPUDisplay(GPUDisplayFrontend* frontend, GPUChainTracking* chain, GPUQA* qa, const char* backend = "opengl", const GPUParam* param = nullptr, const GPUCalibObjectsConst* calib = nullptr, const GPUSettingsDisplay* config = nullptr);
  ~GPUDisplay() = default;
  GPUDisplay(const GPUDisplay&) = delete;

  int StartDisplay();
  void ShowNextEvent(const GPUTrackingInOutPointers* ptrs = nullptr);
  void WaitForNextEvent();
  void SetCollisionFirstCluster(unsigned int collision, int slice, int cluster);

  void HandleKey(unsigned char key);
  int DrawGLScene(bool mixAnimation = false, float mAnimateTime = -1.f);
  void HandleSendKey(int key);
  int InitDisplay(bool initFailure = false);
  void ExitDisplay();
  void ReSizeGLScene(int width, int height, bool init = false);

  const GPUSettingsDisplayRenderer& cfgR() const { return mCfgR; }
  const GPUSettingsDisplayLight& cfgL() const { return mCfgL; }
  int renderWidth() const { return mRenderwidth; }
  int renderHeight() const { return mRenderheight; }
  int screenWidth() const { return mScreenwidth; }
  int screenHeight() const { return mScreenheight; }
  bool useMultiVBO() const { return mUseMultiVBO; }
  const GPUDisplayBackend* backend() const { return mBackend.get(); }
  vecpod<int>* vertexBufferStart() { return mVertexBufferStart; }
  const vecpod<unsigned int>* vertexBufferCount() const { return mVertexBufferCount; }
  struct vtx {
    float x, y, z;
    vtx(float a, float b, float c) : x(a), y(b), z(c) {}
  };
  vecpod<vtx>* vertexBuffer() { return mVertexBuffer; }
  const GPUParam* param() { return mParam; }
  GPUDisplayFrontend* frontend() { return mFrontend; }

 private:
  static constexpr int NSLICES = GPUChainTracking::NSLICES;

  static constexpr const int N_POINTS_TYPE = 15;
  static constexpr const int N_POINTS_TYPE_TPC = 9;
  static constexpr const int N_POINTS_TYPE_TRD = 2;
  static constexpr const int N_POINTS_TYPE_TOF = 2;
  static constexpr const int N_POINTS_TYPE_ITS = 2;
  static constexpr const int N_LINES_TYPE = 7;
  static constexpr const int N_FINAL_TYPE = 4;
  static constexpr int TRACK_TYPE_ID_LIMIT = 100;
  enum PointTypes { tCLUSTER = 0,
                    tINITLINK = 1,
                    tLINK = 2,
                    tSEED = 3,
                    tTRACKLET = 4,
                    tSLICETRACK = 5,
                    tGLOBALTRACK = 6,
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
    vecpod<int> start[N_FINAL_TYPE];
    vecpod<unsigned int> count[N_FINAL_TYPE];
    std::pair<vecpod<int>*, vecpod<unsigned int>*> vBuf[N_FINAL_TYPE];
    threadVertexBuffer() : buffer()
    {
      for (int i = 0; i < N_FINAL_TYPE; i++) {
        vBuf[i].first = start + i;
        vBuf[i].second = count + i;
      }
    }
    void clear()
    {
      for (int i = 0; i < N_FINAL_TYPE; i++) {
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

  int DrawGLScene_internal(bool mixAnimation, float mAnimateTime);
  int InitDisplay_internal();
  int getNumThreads();
  void disableUnsupportedOptions();
  int buildTrackFilter();
  const GPUTPCTracker& sliceTracker(int iSlice);
  const GPUTRDTrackerGPU& trdTracker();
  const GPUTRDGeometry& trdGeometry();
  const GPUTrackingInOutPointers* mIOPtrs = nullptr;
  void insertVertexList(std::pair<vecpod<int>*, vecpod<unsigned int>*>& vBuf, size_t first, size_t last);
  void insertVertexList(int iSlice, size_t first, size_t last);
  template <typename... Args>
  void SetInfo(Args... args)
  {
    snprintf(mInfoText2, 1024, args...);
    GPUInfo("%s", mInfoText2);
    mInfoText2Timer.ResetStart();
  }
  void PrintGLHelpText(float colorValue);
  void calcXYZ(const float*);
  void mAnimationCloseAngle(float& newangle, float lastAngle);
  void mAnimateCloseQuaternion(float* v, float lastx, float lasty, float lastz, float lastw);
  void setAnimationPoint();
  void resetAnimation();
  void removeAnimationPoint();
  void startAnimation();
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
  void SetColorGlobalTracks();
  void SetColorFinal();
  void SetColorGrid();
  void SetColorGridTRD();
  void SetColorMarked();
  void SetCollisionColor(int col);
  void setQuality();
  void setDepthBuffer();
  void setFrameBuffer(int updateCurrent = -1, unsigned int newID = 0);
  void UpdateOffscreenBuffers(bool clean = false);
  void updateConfig();
  void drawPointLinestrip(int iSlice, int cid, int id, int id_limit = TRACK_TYPE_ID_LIMIT);
  vboList DrawClusters(int iSlice, int select, unsigned int iCol);
  vboList DrawSpacePointsTRD(int iSlice, int select, int iCol);
  vboList DrawSpacePointsTOF(int iSlice, int select, int iCol);
  vboList DrawSpacePointsITS(int iSlice, int select, int iCol);
  vboList DrawLinks(const GPUTPCTracker& tracker, int id, bool dodown = false);
  vboList DrawSeeds(const GPUTPCTracker& tracker);
  vboList DrawTracklets(const GPUTPCTracker& tracker);
  vboList DrawTracks(const GPUTPCTracker& tracker, int global);
  void DrawTrackITS(int trackId, int iSlice);
  GPUDisplay::vboList DrawFinalITS();
  template <class T>
  void DrawFinal(int iSlice, int /*iCol*/, GPUTPCGMPropagator* prop, std::array<vecpod<int>, 2>& trackList, threadVertexBuffer& threadBuffer);
  vboList DrawGrid(const GPUTPCTracker& tracker);
  vboList DrawGridTRD(int sector);
  void DoScreenshot(char* filename, float mAnimateTime = -1.f);
  void PrintHelp();
  void createQuaternionFromMatrix(float* v, const float* mat);
  void drawVertices(const vboList& v, const GPUDisplayBackend::drawType t);

  GPUDisplayFrontend* mFrontend = nullptr;
  std::unique_ptr<GPUDisplayBackend> mBackend;
  GPUChainTracking* mChain = nullptr;
  const GPUParam* mParam = nullptr;
  const GPUCalibObjectsConst* mCalib = nullptr;
  const GPUSettingsDisplay& mConfig;
  GPUSettingsDisplayLight mCfgL;
  GPUSettingsDisplayHeavy mCfgH;
  GPUSettingsDisplayRenderer mCfgR;
  GPUQA* mQA;
  qSem mSemLockDisplay;

  int mNDrawCalls = 0;

  bool mUseMultiVBO = false;

  std::array<float, 3> mDrawColor = {};

  int mTestSetting = 0;

  float mAngleRollOrigin = -1e9;
  float mMaxClusterZ = -1;

  int mScreenwidth = GPUDisplayFrontend::INIT_WIDTH, mScreenheight = GPUDisplayFrontend::INIT_HEIGHT;
  int mRenderwidth = GPUDisplayFrontend::INIT_WIDTH, mRenderheight = GPUDisplayFrontend::INIT_HEIGHT;

  hmm_mat4 mViewMatrix, mModelMatrix;
  float* const mViewMatrixP = &mViewMatrix.Elements[0][0];
  float mXYZ[3];
  float mAngle[3];
  float mRPhiTheta[3];
  float mQuat[4];

  vecpod<std::array<int, 37>> mCollisionClusters;
  int mNCollissions = 1;

  vecpod<vtx> mVertexBuffer[NSLICES];
  vecpod<int> mVertexBufferStart[NSLICES];
  vecpod<unsigned int> mVertexBufferCount[NSLICES];

  vecpod<unsigned int> mMainBufferStack{0};
  GPUDisplayBackend::GLfb mMixBuffer;
  GPUDisplayBackend::GLfb mOffscreenBuffer, mOffscreenBufferNoMSAA;

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
  int mNMaxClusters = 0;
  int mNMaxSpacePointsTRD = 0;
  int mNMaxClustersITS = 0;
  int mNMaxClustersTOF = 0;
  int mCurrentClusters = 0;
  int mCurrentSpacePointsTRD = 0;
  int mCurrentClustersITS = 0;
  int mCurrentClustersTOF = 0;
  vecpod<int> mTRDTrackIds;
  vecpod<bool> mITSStandaloneTracks;
  std::vector<bool> mTrackFilter;
  bool mUpdateTrackFilter = false;

  int mGlDLrecent = 0;
  volatile int mUpdateDLList = 0;
  volatile int mResetScene = 0;

  int mAnimate = 0;
  HighResTimer mAnimationTimer;
  int mAnimationFrame = 0;
  int mAnimationLastBase = 0;
  int mAnimateScreenshot = 0;
  int mAnimationExport = 0;
  bool mAnimationChangeConfig = true;
  float mAnimationDelay = 2.f;
  vecpod<float> mAnimateVectors[9];
  vecpod<GPUSettingsDisplayLight> mAnimateConfig;
  opengl_spline mAnimationSplines[8];

  int mPrintInfoText = 1;
  char mInfoText2[1024];
  HighResTimer mInfoText2Timer, mInfoHelpTimer;

  std::vector<threadVertexBuffer> mThreadBuffers;
  std::vector<std::vector<std::array<std::array<vecpod<int>, 2>, NSLICES>>> mThreadTracks;
  volatile int mInitResult = 0;

  float mFPSScale = 1, mFPSScaleadjust = 0;
  int mFramesDone = 0, mFramesDoneFPS = 0;
  HighResTimer mTimerFPS, mTimerDisplay, mTimerDraw;
  vboList mGlDLLines[NSLICES][N_LINES_TYPE];
  vecpod<std::array<vboList, N_FINAL_TYPE>> mGlDLFinal[NSLICES];
  vboList mGlDLFinalITS;
  vecpod<vboList> mGlDLPoints[NSLICES][N_POINTS_TYPE];
  vboList mGlDLGrid[NSLICES];
  vboList mGlDLGridTRD[NSLICES / 2];
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif
