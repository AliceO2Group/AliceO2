/// \file CheckVertexer.C
/// \brief Simple macro to run the Vertexer

#if !defined(__CLING__) || defined(__ROOTCLING__)
  #include <array>
  #include <string>
  #include <vector>
  #include <iostream>
  #include "ITSReconstruction/CA/Event.h"
  #include "ITSReconstruction/CA/vertexer/Vertexer.h"
  #include "ITSReconstruction/CA/IOUtils.h"
#endif

using namespace o2::ITS::CA;

void CheckVertexer(const std::string& fname = "data.txt", const float zCut = 0.02, const float phiCut = 0.005, const int pairCut = -1 )
{
  std::vector<Event> events = IOUtils::loadEventData(fname);
  std::cout<<"Loaded: "<< fname <<" containing "<< events.size() << " events(s)" <<std::endl;
  for ( auto event : events ) {
    Vertexer vertexer(event);
    vertexer.initialise(zCut, phiCut, pairCut);
    // vertexer.computeTriplets();
    vertexer.findTracklets();
    //vertexer.checkTriplets();
  }
}
