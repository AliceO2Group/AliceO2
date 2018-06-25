#ifndef ALICEO2_FIT_RECPOINTS_H
#define ALICEO2_FIT_RECPOINTS_H

#include "CommonDataFormat/TimeStamp.h"
#include <array>
#include "Rtypes.h"
#include <TObject.h>
#include <FITBase/Digit.h>

namespace o2
{
namespace fit
{
class RecPoints
{
 public:
  RecPoints() = default;
  RecPoints(const std::array<Float_t, 3>& collisiontime,
            Float_t vertex,
            std::vector<ChannelData> timeamp)
    : mCollisionTime(collisiontime),
      mVertex(vertex),
      mTimeAmp(std::move(timeamp))
  {
  }
  ~RecPoints() = default;

  //void FillFromDigits(const std::vector<Digit>& digits);
  void FillFromDigits(const Digit* digit);
  Float_t GetCollisionTime(int side) const { return mCollisionTime[side]; }
  void setCollisionTime(Float_t time, int side) { mCollisionTime[side] = time; }

  Float_t GetVertex(Float_t vertex) const { return mVertex; }
  void setVertex(Float_t vertex) { mVertex = vertex; }

  void SetMgrEventTime(Double_t time) {mEventTime=time;}

  const std::vector<ChannelData>& getChDgData() const { return mTimeAmp; }
  void setChDgData(const std::vector<ChannelData>& TimeAmp) { mTimeAmp = TimeAmp; }
  void setChDgData(std::vector<ChannelData>&& TimeAmp) { mTimeAmp = std::move(TimeAmp); }


 private:
  std::array<Float_t, 3> mCollisionTime;
  Float_t mVertex = 0;
  Double_t mEventTime; //event time from Fair for continuous 
  std::vector<ChannelData> mTimeAmp;
  Int_t mEventID = 0;       ///< current event id from the source

  ClassDefNV(RecPoints, 1);
};
} // namespace fit
} // namespace o2
#endif
