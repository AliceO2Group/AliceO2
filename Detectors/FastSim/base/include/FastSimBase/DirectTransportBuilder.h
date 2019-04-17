#ifndef ALICEO2_FASTSIM_BASE_DIRECTTRANSPORTBUILDER_H_
#define ALICEO2_FASTSIM_BASE_DIRECTTRANSPORTBUILDER_H_

#include <G4String.hh>

#include "FastSimBase/DirectTransport.h"

class G4Region;

namespace o2
{
namespace base
{

class DirectTransport;

/// This fast sim model just pushes a track through a FairModule repspecting
/// external fields as well
class DirectTransportBuilder
{
  public:

    static DirectTransportBuilder& Instance()
    {
      static DirectTransportBuilder inst;
      return inst;
    }

    DirectTransport* build(G4Region* region) const;
    DirectTransport* build(const G4String& name) const;

    ~DirectTransportBuilder() = default;

  private:
    DirectTransportBuilder() = default;

};


} // namespace base
} // namespace o2

#endif /* ALICEO2_FASTSIM_BASE_DIRECTTRANSPORTBUILDER_H_ */
