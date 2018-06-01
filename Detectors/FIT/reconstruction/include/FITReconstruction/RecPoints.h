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
struct Channel {
  Int_t index;
  Float_t time, amp;
  ClassDefNV(Channel, 1);
};
class RecPoints
{
 public:
  RecPoints() = default;
  RecPoints(const std::array<Float_t, 3>& collisiontime,
            Float_t vertex,
            std::vector<Channel> timeamp)
    : mCollisionTime(collisiontime),
      mVertex(vertex),
      mTimeAmp(std::move(timeamp))
  {
  }
  ~RecPoints() = default;

  void FillFromDigits(const std::vector<Digit>& digits);
  Float_t GetCollisionTime(int side) const { return mCollisionTime[side]; }
  void setCollisionTime(Float_t time, int side) { mCollisionTime[side] = time; }

  Float_t GetVertex(Float_t vertex) const { return mVertex; }
  void setVertex(Float_t vertex) { mVertex = vertex; }

 private:
  std::array<Float_t, 3> mCollisionTime;
  Float_t mVertex = 0;
  std::vector<Channel> mTimeAmp;

  ClassDefNV(RecPoints, 1);
};
} // namespace fit
} // namespace o2
#endif
