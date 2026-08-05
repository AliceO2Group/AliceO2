/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----               FairGenericVMCConfig header file                -----
// -----            Created 2019.02.19 by R. Karabowicz                -----
// -------------------------------------------------------------------------

#ifndef FAIR_GENERIC_VMC_CONFIG_H
#define FAIR_GENERIC_VMC_CONFIG_H

class FairGenericVMCConfig
{
  public:
    FairGenericVMCConfig();
    virtual ~FairGenericVMCConfig();

    virtual void Setup(const char* mcEngine);
};

#endif
