
// common piece of code to setup stack and register
// with VMC instances
template <typename T, typename R>
void stackSetup(T* vmc, R* run) {
  // create the O2 vmc stack instance
  auto st = new o2::Data::Stack();
  st->setMinHits(1);
  st->StoreSecondaries(kTRUE);
  vmc->SetStack(st);

  // register the stack as an observer on FinishPrimary events (managed by Cave)
  bool foundCave = false;
  auto modules = run->GetListOfModules();
  
  if( strcmp(vmc->GetName(), "TGeant4")==0 ) {
    // there is no way to get the module by name, so we have
    // to iterate through the complete list
    for (auto m : *modules) {
      if(strcmp("CAVE", ((FairModule*)m)->GetName()) == 0) {
        // this thing is the cave
        if(auto c=dynamic_cast<o2::Passive::Cave*>(m)) {
          foundCave = true;
          c->addFinishPrimaryHook([st](){ st->notifyFinishPrimary();});
        }
      }
    }
    if (!foundCave) {
      LOG(FATAL) << "Cave volume not found; Could not attach observers" << FairLogger::endl;
    }
  }
}
