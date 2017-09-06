// sample macro to analyse steps

template <typename T>
void insertorinc(T &value, std::map<T, int>, int c=1) {

}

void insertorinc(int index, std::vector<int> &v, int c=1) {
  if(index >= v.size()) {
	v.resize(index+1, 0);
  }
  v[index]+=c;
}

std::string toString(std::string *s){
	if(s==nullptr || s->size()==0) {
		return "UNKNOWN";
	}
	return *s;
}

// transforms hist of volid to a hist having names
template <typename T>
void volidTovolnameHist(std::vector<T> const &v, std::map<std::string,T> &m, o2::StepLookups *l){
  for(int i=0;i<v.size();++i){
	std::string volname=toString(l->volidtovolname[i]);
    volname.append("/"+toString(l->volidtomodule[i]));
	m.insert(std::pair<std::string,T>(volname, v[i]));
  }
}

template <typename T, typename V=int>
void printSortedMap(std::map<T, V> const &m, std::string const &message, int cut=-1) {
  // accumulate
  V accum=0;
  for(auto &p : m) {
	accum+=p.second;
  }

  // sort after value
  using P=std::pair<T, V>;
  std::vector<P> copy(m.begin(),m.end());
  std::sort(copy.begin(), copy.end(), [](P a, P b){return a.second > b.second;});

  // print
  int counter=0;
  std::cout << "----------- " << message << " ---------------\n";
  std::cout << "TOTAL: " << accum << "\n";
  for(auto &p : copy) {
    if(cut == -1 || counter < cut) {
	  std::cout << p.first << " " << p.second << " " << p.second/(1.*accum) << "\n";
    }
    counter++;
  }
}

template <typename T>
void fetchData( TBranch *br, T** address) {
  br->SetAddress(address);
  br->GetEntry(0);
}

void analyseSteps(const char *filename) {
 // load data
 auto *f = TFile::Open(filename);
 if(!f) return;

 auto *tree=(TTree*)f->Get("StepLoggerTree");

 // the lookup branch
 auto lbranch = tree->GetBranch("Lookups");
 auto sbranch = tree->GetBranch("Steps");
 auto callbranch = tree->GetBranch("Calls");

 o2::StepLookups *lookup =nullptr;
 fetchData(lbranch, &lookup);
 std::cerr << "lookup " << lookup << "\n";

 std::vector<o2::StepInfo> *steps = nullptr;
 fetchData(sbranch, &steps);
 std::cerr << "steps " << steps->size() << "\n";

 std::vector<o2::MagCallInfo> *calls = nullptr;
 fetchData(callbranch, &calls);

 // everything loaded

 std::vector<int> volidhist;
 std::vector<int> secondaryvolidhist; // secondaries per volid
 // loop over the steps data
 for(auto &st : *steps) {
   insertorinc(st.volId, volidhist);
   insertorinc(st.volId, secondaryvolidhist, st.nsecondaries);
 }

 std::map<std::string, int> volnamehist;
 volidTovolnameHist(volidhist, volnamehist, lookup);
 printSortedMap(volnamehist,"STEPS PER VOLUME (sorted)", 30);

 std::map<std::string, int> secondaryvolnamehist;
 volidTovolnameHist(secondaryvolidhist, secondaryvolnamehist, lookup);
 printSortedMap(secondaryvolnamehist,"SECONDARIES PER VOLUME (sorted)", 30);

 // analyse some stuff on calls
 std::vector<int> volidcallhist;
 std::vector<int> voliduselesscallhist;
 for(auto &call : *calls) {
   auto step = steps->operator[](call.stepid);
   insertorinc(step.volId, volidcallhist);
   if (call.B < 0.01) {
	 insertorinc(step.volId, voliduselesscallhist);
   }
 }

 // make a ratio of total to useless field calls
 std::vector<float> callratio(voliduselesscallhist.size(),1.);
 for(int i=0;i<voliduselesscallhist.size();++i) {
   callratio[i]=voliduselesscallhist[i]/(1.*volidcallhist[i]+0.01);
 }

 {
   std::map<std::string, int> tmp;
   volidTovolnameHist(volidcallhist, tmp, lookup);
   printSortedMap(tmp,"FIELD CALLS PER VOLUME (sorted)", 30);
 }

 {
   std::map<std::string, int> tmp;
   volidTovolnameHist(voliduselesscallhist, tmp, lookup);
   printSortedMap(tmp,"USELESS PER VOLUME (sorted)", 30);
 }


 //{
 //  std::map<std::string, float> tmp;
   //volidTovolnameHist(callratio, tmp, lookup);
   //printSortedMap(tmp,"USELESS PER VOLUME (sorted by ratio)", 1000);
 //}


}
