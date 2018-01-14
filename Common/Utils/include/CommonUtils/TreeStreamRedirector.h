// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief Class for creating debug root trees with std::cout like intervace
/// @author Marian Ivanov, marian.ivanov@cern.ch (original code in AliRoot)
///         Ruben Shahoyan, ruben.shahoyan@cern.ch (porting to O2)

#ifndef ALICEO2_TREESTREAMREDIRECTOR_H
#define ALICEO2_TREESTREAMREDIRECTOR_H

#include <Rtypes.h>
#include <TDirectory.h>
#include "CommonUtils/TreeStream.h"

namespace o2
{
namespace utils
{
/// The TreeStreamRedirector class manages one or few TreeStream objects to be written to
/// the same output file.
/// TreeStreamRedirector myTreeStreamRedirector("myOutFile.root","recreate");
/// myTreeStreamRedirector<<"myStream0"<<"brName00="<<obj00<<"brName01="<<obj01<<"\n";
/// ...
/// myTreeStreamRedirector<<"myStream2"<<"brName10="<<obj10<<"brName11="<<obj11<<"\n";
/// ...
/// will create ouput file with 2 trees stored.
///
/// The flushing of trees to the file happens on TreeStreamRedirector::Close() call
/// or at its desctruction.
/// 
/// See testTreeStream.cxx for functional example
///  
class TreeStreamRedirector
{
 public:
  TreeStreamRedirector(const char* fname = "", const char* option = "update");
  virtual ~TreeStreamRedirector();
  void Close();
  TFile* GetFile() { return mDirectory->GetFile(); }
  TDirectory* GetDirectory() { return mDirectory; }
  virtual TreeStream& operator<<(Int_t id);
  virtual TreeStream& operator<<(const char* name);
  void SetDirectory(TDirectory* sfile);
  void SetFile(TFile* sfile);
  static void FixLeafNameBug(TTree* tree);

 private:
  TreeStreamRedirector(const TreeStreamRedirector& tsr);
  TreeStreamRedirector& operator=(const TreeStreamRedirector& tsr);

  std::unique_ptr<TDirectory> mOwnDirectory;             // own directory of the redirector
  TDirectory* mDirectory = nullptr;                      // output directory
  std::vector<std::unique_ptr<TreeStream>> mDataLayouts; // array of data layouts

  ClassDefNV(TreeStreamRedirector, 0);
};
}
}

#endif
