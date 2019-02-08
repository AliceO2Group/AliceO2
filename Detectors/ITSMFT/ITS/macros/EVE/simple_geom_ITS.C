// Based on a macro by Jeremi Niedziela on 09/02/2016.

#if !defined(__CINT__) || defined(__MAKECINT__)

#include <TGeoManager.h>
#include <TGeoNode.h>
#include <TEveManager.h>
#include <TEveElement.h>
#include <TEveGeoNode.h>
#include <TSystem.h>

#include <TObjString.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#endif

void AddNodes(TGeoNode* node, TEveGeoNode* parent, Int_t depth, Int_t depthmax, TObjArray* list);

void simple_geom_ITS(std::string inputGeom = "O2geometry.root")
{
  const char* currentDetector = "ITS";

  // load geometry library
  gSystem->Load("libGeom");

  // create visualisation manager
  TEveManager::Create();

  TGeoManager::Import(inputGeom.c_str());

  // find main node for our detector
  TGeoNode* tnode = gGeoManager->GetTopNode();
  tnode->SetVisibility(kFALSE);

  TEveGeoTopNode* eve_tnode = new TEveGeoTopNode(gGeoManager, tnode);
  eve_tnode->SetVisLevel(0);

  gEve->AddGlobalElement(eve_tnode);

  std::string listName("geom_list_");
  listName += currentDetector;
  listName += ".txt";
  ifstream in(listName, ios::in);
  std::cout << "Adding shapes from file:" << listName << '\n';

  int lineIter = 0;
  while (true) {
    std::string line;
    in >> line;
    if (in.eof())
      break;

    if (line[0] == '#')
      continue;

    TString path(line);

    if (!path.Contains("cave"))
      continue;

    TObjArray* list = path.Tokenize("/");
    Int_t depth = list->GetEntries();
    AddNodes(tnode, eve_tnode, depth, depth, list);
    lineIter++;
  }
  in.close();

  if (lineIter == 0) {
    std::cout << "File for " << currentDetector << " is empty. Skipping...\n";
  } else {
    std::string fname("simple_geom_");
    fname += currentDetector;
    fname += ".root";
    eve_tnode->SaveExtract(fname.c_str(), currentDetector, kTRUE);
  }
}

void AddNodes(TGeoNode* node, TEveGeoNode* parent, Int_t depth, Int_t depthmax, TObjArray* list)
{
  if (--depth <= 0)
    return;

  TObjArray* nlist = node->GetVolume()->GetNodes(); // all nodes in current level
  if (!nlist)
    return;

  TObjString* nname = (TObjString*)list->At(depthmax - depth); // name of required node in current level

  for (int i = 0; i < nlist->GetEntries(); i++) { // loop over nodes in current level and find the one with matching name
    TGeoNode* node2 = (TGeoNode*)nlist->At(i);

    if (strcmp(node2->GetName(), nname->GetString().Data()) == 0) {
      TEveGeoNode* son = dynamic_cast<TEveGeoNode*>(parent->FindChild(nname->GetName()));
      if (!son) {
        son = new TEveGeoNode(node2);
        parent->AddElement(son);
      }
      AddNodes(node2, son, depth, depthmax, list);
    }
  }
}
