#include <G4Region.hh>

#include "FastSimBase/DirectTransport.h"
#include "FastSimBase/DirectTransportBuilder.h"

using namespace o2::base;

DirectTransport* DirectTransportBuilder::build(G4Region* region) const
{
  // New direct transport model with the G4Region just extracted.
  return new DirectTransport(region->GetName() + "_directTransport", region);
}

DirectTransport* DirectTransportBuilder::build(const G4String& name) const
{
  // New direct transport model with the G4Region just extracted.
  return new DirectTransport(name + "_directTransport");
}
