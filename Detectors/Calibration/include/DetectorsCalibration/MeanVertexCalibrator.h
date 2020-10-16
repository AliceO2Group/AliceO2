// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MEAN_VERTEX_CALIBRATOR_H_
#define MEAN_VERTEX_CALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DetectorsCalibration/MeanVertexData.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "CCDB/CcdbObjectInfo.h"
#include <array>
#include <deque>

namespace o2
{
namespace calibration
{

class MeanVertexCalibrator final : public o2::calibration::TimeSlotCalibration<o2::dataformats::PrimaryVertex, o2::calibration::MeanVertexData>
{
  using PVertex = o2::dataformats::PrimaryVertex;
  using MeanVertexData = o2::calibration::MeanVertexData;
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<MeanVertexData>;
  using MVObject = o2::dataformats::MeanVertexObject;
  using MVObjectVector = std::vector<MVObject>;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;

 public:
 MeanVertexCalibrator(int minEnt = 500, bool useFit = false, int nBinsX = 1000, float rangeX = 10.f,
		      int nBinsY = 1000, float rangeY = 10.f, int nBinsZ = 1000, float rangeZ = 10.f,
		      int nSlotsSMA = 5) :
  mMinEntries(minEnt), mUseFit(useFit), mNBinsX(nBinsX), mRangeX(rangeX), mNBinsY(nBinsY), mRangeY(rangeY),
    mNBinsZ(nBinsZ), mRangeZ(rangeZ), mSMAslots(nSlotsSMA) {
    mSMAdata.init(useFit, nBinsX, rangeX, nBinsY, rangeY, nBinsZ, rangeZ);
  }

  ~MeanVertexCalibrator() final = default;

  bool hasEnoughData(const Slot& slot) const final { LOG(INFO) << "container entries = " << slot.getContainer()->entries << ", minEntries = " << mMinEntries; return slot.getContainer()->entries >= mMinEntries; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  uint64_t getNSlotsSMA() const { return mSMAslots; }
  void setNSlotsSMA(uint64_t nslots) { mSMAslots = nslots; }

  void doSimpleMovingAverage(std::deque<float>& dq, float& sma);

  const MVObjectVector& getMeanVertexObjectVector() const { return mMeanVertexVector; }
  const CcdbObjectInfoVector& getMeanVertexObjectInfoVector() const { return mInfoVector; }
  CcdbObjectInfoVector& getMeanVertexObjectInfoVector() { return mInfoVector; }

 private:
  int mMinEntries = 0;
  int mNBinsX = 0;
  float mRangeX = 0.;
  int mNBinsY = 0;
  float mRangeY = 0.;
  int mNBinsZ = 0;
  float mRangeZ = 0.;
  bool mUseFit = false;
  uint64_t mSMAslots = 5; 
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description
                                    // of the accompanying LHCPhase
  MVObjectVector mMeanVertexVector;   // vector of Mean Vertex Objects, each element is filled in "process"
                                      // when we finalize one slot (multiple can be finalized during the same
                                      // "process", which is why we have a vector. Each element is to be considered
                                      // the output of the device, and will go to the CCDB. It is the simple
                                      // moving average
  std::deque<float> mTmpMVobjDqX;            // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, X
  std::deque<float> mTmpMVobjDqY;            // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, Y
  std::deque<float> mTmpMVobjDqZ;            // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, Z
  std::deque<float> mTmpMVobjDqSigmaX;       // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, SigmaX
  std::deque<float> mTmpMVobjDqSigmaY;       // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, SigmaY
  std::deque<float> mTmpMVobjDqSigmaZ;       // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, SigmaZ
  float mSMAx;                               // simple moving average, X
  float mSMAy;                               // simple moving average, Y
  float mSMAz;                               // simple moving average, Z
  float mSMAsigmax;                          // simple moving average, sigmaX
  float mSMAsigmay;                          // simple moving average, sigmaY
  float mSMAsigmaz;                          // simple moving average, sigmaZ
  std::deque<TFType> mTmpMVobjDqTimeStart;   // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, start time of used TFs
  std::deque<TFType> mTmpMVobjDqTimeEnd;     // This is the deque of MeanVertex objecs that will be used for the
                                             // simple moving average, end time of used TFs
  std::deque<MeanVertexData> mTmpMVdataDq;   // This is the vector of Mean Vertex data to be used for the simple
                                             // moving average
  MeanVertexData mSMAdata;                   // This is to do the SMA when we keep the histos
  
  ClassDefOverride(MeanVertexCalibrator, 1);
};

} // end namespace calibration
} // end namespace o2

#endif /* TOF_LHCPHASE_CALIBRATION_H_ */
