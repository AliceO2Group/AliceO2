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

 private:
  std::array<Float_t, 3> mCollisionTime;
  Float_t mVertex = 0;
  std::vector<ChannelData> mTimeAmp;

  ClassDefNV(RecPoints, 1);
};
} // namespace fit
} // namespace o2
#endif
