#pragma once

class TObject;

namespace o2
{
namespace qc
{
class Producer
{
 public:
  virtual TObject* produceData() const = 0;
};
}
}