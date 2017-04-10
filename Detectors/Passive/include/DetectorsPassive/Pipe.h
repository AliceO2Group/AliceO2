/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliPipe file                                -----
// -----                Created by M. Al-Turany  June 2014             -----
// -------------------------------------------------------------------------

#ifndef ALICEO2_PASSIVE_PIPE_H
#define ALICEO2_PASSIVE_PIPE_H

#include "FairModule.h"  // for FairModule
#include "Rtypes.h"      // for Pipe::Class, ClassDef, Pipe::Streamer

namespace o2 {
namespace Passive {
 

class Pipe : public FairModule {
  public:
    Pipe(const char * name, const char *Title="Alice Pipe");
    Pipe();

    ~Pipe() override;
    void ConstructGeometry() override;

    /// Clone this object (used in MT mode only)
    FairModule* CloneModule() const override;

  private:
    Pipe(const Pipe& orig);
    Pipe& operator=(const Pipe&);
   
  ClassDefOverride(Pipe,1) //PIPE

};
}
}
#endif //PIPE_H

