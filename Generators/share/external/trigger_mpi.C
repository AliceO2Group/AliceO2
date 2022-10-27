// MPI trigger
//
//   usage: o2sim --trigger external --configKeyValues 'TriggerExternal.fileName=trigger_mpi.C;TriggerExternal.funcName="trigger_mpi()"'
//

/// \author R+Preghenella - February 2020

#include "Generators/Trigger.h"
#include "Pythia8/Pythia.h"
#include "TPythia6.h"
#include <fairlogger/Logger.h>

o2::eventgen::DeepTrigger
  trigger_mpi(int mpiMin = 15)
{
  return [mpiMin](void* interface, std::string name) -> bool {
    int nMPI = 0;
    if (!name.compare("pythia8")) {
      auto py8 = reinterpret_cast<Pythia8::Pythia*>(interface);
      nMPI = py8->info.nMPI();
    } else if (!name.compare("pythia6")) {
      auto py6 = reinterpret_cast<TPythia6*>(interface);
      nMPI = py6->GetMSTI(31);
    } else
      LOG(fatal) << "Cannot define MPI for generator interface \'" << name << "\'";
    return nMPI >= mpiMin;
  };
}
