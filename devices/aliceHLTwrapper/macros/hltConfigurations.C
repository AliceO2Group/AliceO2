//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or	     *
//* (at your option) any later version.					     *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   hltConfigurations.C
//  @author Matthias Richter
//  @since  2014-05-20 
//  @brief  Various helper configurations for the ALICE HLT

/**
 * Usage: aliroot -b -q -l \
 *     hltConfigurations.C \
 *     recraw-local.C'("file", "cdb", minEvent, maxEvent, modules)'
 *
 * Macro defines the following configurations:
 * - cluster-collection: emulation of TPC HLT clusters from TPC raw data
 */
void hltConfigurations()
{
  // init the HLT system
  AliHLTSystem* pHLT=AliHLTPluginBase::GetInstance();

  ///////////////////////////////////////////////////////////////////////////////////////////
  //
  // list of configurations
  //
  ///////////////////////////////////////////////////////////////////////////////////////////
  int iMinSlice=0; 
  int iMaxSlice=35;
  int iMinPart=0;
  int iMaxPart=5;

  TString collectionInput;
  for (int slice=iMinSlice; slice<=iMaxSlice; slice++) {
    for (int part=iMinPart; part<=iMaxPart; part++) {
      TString arg;
      TString filterid;
      filterid.Form("TPC-CLFilter_%02d_%d", slice, part);
      arg.Form("-dataspec 0x%02x%02x%02x%02x", slice, slice, part, part);
      AliHLTConfiguration clusterfilter(filterid.Data(), "BlockFilter","TPC-ClusterTransformation", arg.Data());

      TString cid;
      cid.Form("TPC-CLWriter_%02d_%d", slice, part);
      arg=("-directory emulated-tpc-clusters -subdir -specfmt=_0x%08x -blocknofmt=");
      arg+=Form(" -publisher-conf emulated-tpc-clusters_0x%02x%02x%02x%02x.txt", slice, slice, part, part);
      AliHLTConfiguration clusterwriter(cid.Data(), "FileWriter", filterid.Data(), arg.Data());

      if (collectionInput.Length()>0) collectionInput+=" ";
      collectionInput+=cid;
    }
  }
  liHLTConfiguration clustercollection("cluster-collection", "BlockFilter", collectionInput.Data(), "");
}
