// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CommonDataProcessors.h"

#include "Framework/AlgorithmSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/InitContext.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Variant.h"
#include "../../../Algorithm/include/Algorithm/HeaderStack.h"
#include "Framework/OutputObjHeader.h"

#include "TFile.h"
#include "TTree.h"

#include <ROOT/RSnapshotOptions.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include <ROOT/RVec.hxx>
#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>

using namespace o2::framework::data_matcher;

namespace o2
{
namespace framework
{

struct InputObjectRoute {
  std::string uniqueId;
  std::string directory;
  OutputObjHandlingPolicy policy;
  bool operator<(InputObjectRoute const& other) const
  {
    return this->uniqueId < other.uniqueId;
  }
};

struct InputObject {
  TClass* kind = nullptr;
  void* obj = nullptr;
  std::string name;
};

const static std::unordered_map<OutputObjHandlingPolicy, std::string> ROOTfileNames = {{OutputObjHandlingPolicy::AnalysisObject, "AnalysisResults.root"},
                                                                                       {OutputObjHandlingPolicy::QAObject, "QAResults.root"}};


void CommonDataProcessors::table2tree(TTree* tout, std::shared_ptr<arrow::Table> table)
{
  
  // loop over the columns
  for (int ii=0; ii<table->num_columns(); ii++)
  {                                                                              
    // get column information
    TBranch *br = nullptr;
    auto col = table->column(ii);
    const char *cname = col->name().c_str();
    auto chs = col->data()->chunks();
    LOG(DEBUG) << "number of chunks " << chs.size();
    LOG(DEBUG) << " column type " << col->type()->ToString();

    // what follows is an ugly switch
    // the cases cover the different data types of the branches
    auto cdt = col->type()->id();
    switch (cdt) {
    
    case arrow::Type::type::FLOAT:
      {
        LOG(DEBUG) << "case FLOAT";
        Float_t dbuf;
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        { 
          auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::FloatType>>(ch);   
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            // the following if-clause is needed to have all branches filled properly
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }

    case arrow::Type::type::DOUBLE:
      {
        LOG(DEBUG) << "case DOUBLE";
        Double_t dbuf;                                                   
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        {
          auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(ch);
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }

    case arrow::Type::type::UINT16:
      {
        LOG(DEBUG) << "case UINT";
        UShort_t dbuf;                                                   
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        {
          auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt16Type>>(ch);
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }

    case arrow::Type::type::UINT32:
      {
        LOG(DEBUG) << "case UINT";
        UInt_t dbuf = UInt_t();
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        {                                                                                  
auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt32Type>>(ch);
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }

    case arrow::Type::type::UINT64:
      {
        LOG(DEBUG) << "case UINT64";
        ULong64_t dbuf = ULong64_t();
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        {
          auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt64Type>>(ch);  
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }

    case arrow::Type::type::INT16:
      {
        LOG(DEBUG) << "case INT16";
        Short_t dbuf = Short_t();
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        {
          auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int16Type>>(ch);
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }

    case arrow::Type::type::INT32:
      {
        LOG(DEBUG) << "case INT";
        Int_t dbuf = Int_t();
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        {
          auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ch);
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }

    case arrow::Type::type::INT64:
      {
        LOG(DEBUG) << "case INT64";
        Long64_t dbuf =  Long64_t();
        br = tout->Branch(cname,&dbuf);                             
        for(auto const& ch: chs)
        {
          auto p = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(ch);
          for (int jj=0; jj<ch->length(); jj++)
          {
            dbuf = p->Value(jj);
            if (ii==0) { tout->Fill(); } else { br->Fill(); }
          }
        }
        break;
      }
      
    default:
      std::string strerr = std::string("Unsupported data type ")+col->type()->name();
      throw std::runtime_error(strerr.c_str());
      break;
      
    }
    tout->Write("", TObject::kOverwrite);
  }

}


// =============================================================================
// class datait is a data iterator
// it basically allows to iterate over the elements of an arrow::table::Column using datait::next()
// and is used in CommonDataProcessors::table2treeC
class datait
{

  private:
  arrow::ArrayVector chs;   // chunks
  Int_t nchs;               // number of chunks
  Int_t ich;                // chunk counter
  Int_t nr;                 // number of rows
  Int_t ir;                 // row counter

  // data buffers for each data type
  arrow::Type::type dt;
  void *v       = nullptr;
  Float_t  *vF  = nullptr;
  Double_t *vD  = nullptr;
  UShort_t *vs  = nullptr;
  UInt_t *vi    = nullptr;
  ULong64_t *vl = nullptr;
  Short_t *vS   = nullptr;
  Int_t *vI     = nullptr;
  Long64_t *vL  = nullptr;


  // initialize chunk ib
  void dbufini(Int_t ib)
  {
    // get next chunk of given data type dt
    switch (dt)
    {
      case arrow::Type::type::FLOAT:
        vF = (Float_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::FloatType>>(chs.at(ib))->raw_values();
        break;
      case arrow::Type::type::DOUBLE:
        vD = (Double_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(chs.at(ib))->raw_values();
        break;
      case arrow::Type::type::UINT16:
        vs = (UShort_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt16Type>>(chs.at(ib))->raw_values();
        break;
      case arrow::Type::type::UINT32:
        vi = (UInt_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt32Type>>(chs.at(ib))->raw_values();
        break;
      case arrow::Type::type::UINT64:
        vl = (ULong64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt64Type>>(chs.at(ib))->raw_values();
        break;
      case arrow::Type::type::INT16:
        vS = (Short_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int16Type>>(chs.at(ib))->raw_values();
        break;
      case arrow::Type::type::INT32:
        vI = (Int_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(chs.at(ib))->raw_values();
        break;
      case arrow::Type::type::INT64:
        vL = (Long64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(chs.at(ib))->raw_values();
        break;
    }
    // reset number of rows nr and row counter ir
    nr = chs.at(ib)->length();
    ir = 0;
  };
    
  
  public:  
  datait()
  {
    nchs = 0;
    ich  = 0;
    nr   = 0;
    ir   = 0;

  };
  datait(std::shared_ptr<arrow::Column> col)
  {
    dt   = col->type()->id();
    chs  = col->data()->chunks();
    nchs = chs.size();
    
    ich  = 0;
    dbufini(ich);
    LOG(DEBUG) << "datait: " << col->type()->ToString() << " with " << nchs << " chunks";

  };
  ~datait();

  // allows to reinitialize the datait with new column
  void Init(std::shared_ptr<arrow::Column> col)
  {
    dt   = col->type()->id();
    chs  = col->data()->chunks();
    nchs = chs.size();
    
    ich  = 0;
    dbufini(ich);
  }
  
  // increments data buffer pointer or ...
  // returns nullptr if end of data buffer is reached
  void* next()
  {

    // ich and ir contain the current chunk and row
    // return the next element if available
    if (ir>=nr)
    {
      ich++;
      if (ich<nchs)
      {
        dbufini(ich);
      } else {
        // end of data buffer reached
        return (void*) nullptr;
      }
    }
    switch (dt)
    {
      case arrow::Type::type::FLOAT:
        v = (void*)vF++;
        break;
      case arrow::Type::type::DOUBLE:
        v = (void*)vD++;
        break;
      case arrow::Type::type::UINT16:
        v = (void*)vs++;
        break;
      case arrow::Type::type::UINT32:
        v = (void*)vi++;
        break;
      case arrow::Type::type::UINT64:
        v = (void*)vl++;
        break;
      case arrow::Type::type::INT16:
        v = (void*)vS++;
        break;
      case arrow::Type::type::INT32:
        v = (void*)vI++;
        break;
      case arrow::Type::type::INT64:
        v = (void*)vL++;
        break;
    }
    ir++;
    
    return v;
    
  };
    
};

// .............................................................................
void CommonDataProcessors::table2treeC(TTree* tout,
                                       std::shared_ptr<arrow::Table> table,
                                       bool tupdate)
{

  // first create a vector of pairs to hold the column definitions
  // consisting of a TBranch and the respective data iterator
  std::vector<std::pair<TBranch*, datait*>> specbuf;

  // loop over columns
  char *dbuf;
  std::string leaflist;

  LOG(DEBUG) << "table2treeC: " << table->num_columns();
  for (int ii=0; ii<table->num_columns(); ii++)
  {

    auto col = table->column(ii);
    const char *cname = col->name().c_str();
  
    // for each column get a pair of {branch, data iterator}
    // fill the pairs into a std::vector

    auto cdt = col->type()->id();
    switch (cdt) {
      case arrow::Type::type::FLOAT:
        leaflist = col->name()+"/F";
        break;
      case arrow::Type::type::DOUBLE:
        leaflist = col->name()+"/D";
        break;
      case arrow::Type::type::UINT16:
        leaflist = col->name()+"/s";
        break;
      case arrow::Type::type::UINT32:
        leaflist = col->name()+"/i";
        break;
      case arrow::Type::type::UINT64:
        leaflist = col->name()+"/l";
        break;
      case arrow::Type::type::INT16:
        leaflist = col->name()+"/S";
        break;
      case arrow::Type::type::INT32:
        leaflist = col->name()+"/I";
        break;
      case arrow::Type::type::INT64:
        leaflist = col->name()+"/L";
        break;
    }
    
    // two cases
    // tupdate==true:
    //    tree is updated and it is exptected that the branches exists
    // tupdate==false:
    //    the branches are created
    TBranch *br = nullptr;
    if (tupdate)
    {
      br = tout->GetBranch(cname);
    } else {
      br = tout->Branch(cname,dbuf,leaflist.c_str());
    }
    if (!br)
      throw std::runtime_error("Branch does not exist!");

    datait *dit = new datait(col);
    specbuf.push_back(std::pair<TBranch*, datait*>(br, dit));
  
  }
  LOG(INFO) << "table2treeC: size of specbuf " << specbuf.size();
  
  
  // with this loop over the columns again as long as togo is true.
  // togo is set false when any of the data iterators returns nullptr
  void *add = nullptr;
  bool togo = true;
  while (togo)
  {
    for (int ii=0; ii<table->num_columns(); ii++)
    {
      //  SetBufferAddress using data iterator
      if (add = (void*)std::get<1>(specbuf.at(ii))->next())
      {
        // LOG(DEBUG) << "filling branch " << std::get<0>(specbuf.at(ii))->GetName();
        tout->SetBranchAddress(std::get<0>(specbuf.at(ii))->GetName(),add);
      } else {
        togo = false;
        break;
      };
    }

    //  update the tree
    if (togo) tout->Fill();
   
  }
  // write the tree
  tout->Write("", TObject::kOverwrite);

}


// =============================================================================
DataProcessorSpec CommonDataProcessors::getOutputObjSink(outputObjMap const& outMap)
{
  auto writerFunction = [outMap](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto& callbacks = ic.services().get<CallbackService>();
    auto outputObjects = std::make_shared<std::map<InputObjectRoute, InputObject>>();

    auto endofdatacb = [outputObjects](EndOfStreamContext& context) {
      LOG(INFO) << "Writing merged objects to file";
      std::string currentDirectory = "";
      std::string currentFile = "";
      TFile* f[OutputObjHandlingPolicy::numPolicies];
      for (auto i = 0u; i < OutputObjHandlingPolicy::numPolicies; ++i) {
        f[i] = nullptr;
      }
      for (auto& [route, entry] : *outputObjects) {
        auto file = ROOTfileNames.find(route.policy);
        if (file != ROOTfileNames.end()) {
          auto filename = file->second;
          if (f[route.policy] == nullptr) {
            f[route.policy] = TFile::Open(filename.c_str(), "RECREATE");
          }
          auto nextDirectory = route.directory;
          if ((nextDirectory != currentDirectory) || (filename != currentFile)) {
            if (!f[route.policy]->FindKey(nextDirectory.c_str())) {
              f[route.policy]->mkdir(nextDirectory.c_str());
            }
            currentDirectory = nextDirectory;
            currentFile = filename;
          }
          (f[route.policy]->GetDirectory(currentDirectory.c_str()))->WriteObjectAny(entry.obj, entry.kind, entry.name.c_str());
        }
      }
      for (auto i = 0u; i < OutputObjHandlingPolicy::numPolicies; ++i) {
        if (f[i] != nullptr) {
          f[i]->Close();
        }
      }
      LOG(INFO) << "All outputs merged in their respective target files";
      context.services().get<ControlService>().readyToQuit(QuitRequest::All);
    };

    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);
    return [outputObjects, outMap](ProcessingContext& pc) mutable -> void {
      auto const& ref = pc.inputs().get("x");
      if (!ref.header) {
        LOG(ERROR) << "Header not found";
        return;
      }
      if (!ref.payload) {
        LOG(ERROR) << "Payload not found";
        return;
      }
      auto datah = o2::header::get<o2::header::DataHeader*>(ref.header);
      if (!datah) {
        LOG(ERROR) << "No data header in stack";
        return;
      }

      auto objh = o2::header::get<o2::framework::OutputObjHeader*>(ref.header);
      if (!objh) {
        LOG(ERROR) << "No output object header in stack";
        return;
      }

      FairTMessage tm(const_cast<char*>(ref.payload), datah->payloadSize);
      InputObject obj;
      obj.kind = tm.GetClass();
      if (obj.kind == nullptr) {
        LOGP(error, "Cannot read class info from buffer.");
        return;
      }

      OutputObjHandlingPolicy policy = OutputObjHandlingPolicy::AnalysisObject;
      if (objh)
        policy = objh->mPolicy;

      obj.obj = tm.ReadObjectAny(obj.kind);
      TNamed* named = static_cast<TNamed*>(obj.obj);
      obj.name = named->GetName();
      auto lookup = outMap.find(obj.name);
      std::string directory{"VariousObjects"};
      if (lookup != outMap.end()) {
        directory = lookup->second;
      }
      InputObjectRoute key{obj.name, directory, policy};
      auto existing = outputObjects->find(key);
      if (existing == outputObjects->end()) {
        outputObjects->insert(std::make_pair(key, obj));
        return;
      }
      auto merger = existing->second.kind->GetMerge();
      if (!merger) {
        LOGP(error, "Already one object found for {}.", obj.name);
        return;
      }

      TList coll;
      coll.Add(static_cast<TObject*>(obj.obj));
      merger(existing->second.obj, &coll, nullptr);
    };
  };

  DataProcessorSpec spec{
    "internal-dpl-global-analysis-file-sink",
    {InputSpec("x", DataSpecUtils::dataDescriptorMatcherFrom(header::DataOrigin{"ATSK"}))},
    Outputs{},
    AlgorithmSpec(writerFunction),
    {}};

  return spec;
}


// add sink for dangling AODs 
DataProcessorSpec
  CommonDataProcessors::getGlobalAODSink(std::vector<InputSpec> const& danglingOutputInputs)
{

  auto writerFunction = [danglingOutputInputs](InitContext& ic) -> std::function<void(ProcessingContext&)>
  {
    
    LOG(DEBUG) << "======== getGlobalAODSink::Inint ==========";
    
    // analyze ic and take actions accordingly
    auto fnbase     = ic.options().get<std::string>("res-file");
    auto filemode   = ic.options().get<std::string>("res-mode");
    auto keepString = ic.options().get<std::string>("keep");
    auto ntfmerge   = ic.options().get<Int_t>("ntfmerge");
    
    // find out if any tables need to be saved
    bool hasOutputsToWrite = true;
    bool usematch          = false;
    
    
    // use the parameter keep to create a matcher
    std::shared_ptr<data_matcher::DataDescriptorMatcher> matcher;
    if (!keepString.empty())
    {
      usematch = true;
      
      auto [variables, outputMatcher] = DataDescriptorQueryBuilder::buildFromKeepConfig(keepString);
      matcher = outputMatcher;
      VariableContext context;
      for (auto& spec : danglingOutputInputs) {
        auto concrete = DataSpecUtils::asConcreteDataMatcher(spec);
        if (outputMatcher->match(concrete, context)) {
          hasOutputsToWrite = true;
        }
      }
    }

    // if nothing needs to be saved then return a trivial functor
    if (!hasOutputsToWrite) {
      return std::move([](ProcessingContext& pc) mutable -> void {
        static bool once = false;
        if (!once) {
          LOG(DEBUG) << "No dangling output to be saved.";
          once = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      });
    }
    
    // end of data functor is called at the end of the data stream
    auto fout = std::make_shared<TFile>();
    auto endofdatacb = [fout](EndOfStreamContext& context) {
      
      if (fout)
        fout->Close();
      
      context.services().get<ControlService>().readyToQuit(QuitRequest::All);
    
    };

    auto& callbacks = ic.services().get<CallbackService>();
    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);
 

    // this functor is called once per time frame
    Int_t ntf = 0;
    return std::move([fout,fnbase,filemode,ntf,ntfmerge,usematch,matcher](ProcessingContext& pc) mutable -> void {
      LOG(DEBUG) << "======== getGlobalAODSink::processing ==========";
      LOG(DEBUG) << " processing data set with " << pc.inputs().size() << " entries";
      LOG(DEBUG) << " result are saved to " << fnbase << "_*.root";
      
      // return immediately if pc.inputs() is empty
      auto ninputs = pc.inputs().size();
      if (ninputs==0) {
        LOG(DEBUG) << "no inputs available!";
        return;
      }
      
      // new strategy since RDataFrame::Snapshot does not work
      //
      // loop over all inputs and extract the arrow::tables
      // create a tree for each arrow::table
      // loop over the columns of a table
      // for each column create a branch
      // loop over the rows of the column
      // fill the values into the corresponding branches
      // write the table
      // see table2treeC
      
      // open new file if ntfmerge time frames is reached
      LOG(INFO) << "This is time frame number "<< ntf;
      bool tupdate = true;
      std::string fname;
      if ((ntf%ntfmerge)==0)
      {
        if (fout) fout->Close();
        
        fname = fnbase+"_"+std::to_string((Int_t)(ntf/ntfmerge))+".root";
        fout =
          std::make_shared<TFile>(fname.c_str(),filemode.c_str());
        tupdate = false;
      }
      ntf++;
      
      // loop over the DataRefs which are contained in pc.inputs()
      TTree *tout = nullptr;
      VariableContext matchingContext;
      for (const auto& ref : pc.inputs()) {
        
        // is this table to be saved?
        auto dh = DataRefUtils::getHeader<header::DataHeader*>(ref);
        // only arrow tables are processed here
        if (dh->payloadSerializationMethod != o2::header::gSerializationMethodArrow)
          continue;

        // does it match the keep parameter
        if (usematch && !matcher->match(*dh, matchingContext))
          continue;


        // get the table name
        auto treename = dh->dataDescription.as<std::string>();
    
        // get the TableConsumer and convert it into an arrow table
        auto s = pc.inputs().get<TableConsumer>(ref.spec->binding);
        auto table = s->asArrowTable();
        LOG(DEBUG) << "The tree name is " << treename; 
        LOG(DEBUG) << "Number of columns " << table->num_columns(); 
        LOG(DEBUG) << "Number of rows     " << table->num_rows();
        
        // we need finite number of rows and columns
        if (table->num_columns()==0 || table->num_rows()==0)
          continue;

        
        // this table needs to be saved to file
        // create the corresponding tree
        if (tupdate)
        {
          tout = (TTree*)fout->Get(treename.c_str());
        } else {
          tout = new TTree(treename.c_str(),treename.c_str());
        }
    
        // write the table to the TTree
        table2treeC(tout,table,tupdate);
        tout->Print();

      }
      
    });  
  };      // end of writerFunction


  DataProcessorSpec spec{
    "internal-dpl-AOD-writter",
    danglingOutputInputs,
    Outputs{},
    AlgorithmSpec(writerFunction),
    {{"res-file", VariantType::String, "AnalysisResults", {"Name of the output file"}},
     {"res-mode", VariantType::String, "RECREATE",        {"Creation mode of the result file: NEW, CREATE, RECREATE, UPDATE"}},
     {"ntfmerge", VariantType::Int,     10,               {"number of time frames to merge into one file"}},
     {"keep",     VariantType::String,  "",               {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION to save in outfile"}}}
  };

  return spec;
}


DataProcessorSpec
  CommonDataProcessors::getGlobalFileSink(std::vector<InputSpec> const& danglingOutputInputs,
                                          std::vector<InputSpec>& unmatched)
{
  auto writerFunction = [danglingOutputInputs](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto filename = ic.options().get<std::string>("outfile");
    auto keepString = ic.options().get<std::string>("keep");

    if (filename.empty()) {
      throw std::runtime_error("output file missing");
    }

    bool hasOutputsToWrite = false;
    auto [variables, outputMatcher] = DataDescriptorQueryBuilder::buildFromKeepConfig(keepString);
    VariableContext context;
    for (auto& spec : danglingOutputInputs) {
      auto concrete = DataSpecUtils::asConcreteDataMatcher(spec);
      if (outputMatcher->match(concrete, context)) {
        hasOutputsToWrite = true;
      }
    }
    if (hasOutputsToWrite == false) {
      return std::move([](ProcessingContext& pc) mutable -> void {
        static bool once = false;
        /// We do it like this until we can use the interruptible sleep
        /// provided by recent FairMQ releases.
        if (!once) {
          LOG(DEBUG) << "No dangling output to be dumped.";
          once = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      });
    }
    auto output = std::make_shared<std::ofstream>(filename.c_str(), std::ios_base::binary);
    return std::move([output, matcher = outputMatcher](ProcessingContext& pc) mutable -> void {
      VariableContext matchingContext;
      LOG(DEBUG) << "processing data set with " << pc.inputs().size() << " entries";
      for (const auto& entry : pc.inputs()) {
        LOG(DEBUG) << "  " << *(entry.spec);
        auto header = DataRefUtils::getHeader<header::DataHeader*>(entry);
        auto dataProcessingHeader = DataRefUtils::getHeader<DataProcessingHeader*>(entry);
        if (matcher->match(*header, matchingContext) == false) {
          continue;
        }
        output->write(reinterpret_cast<char const*>(header), sizeof(header::DataHeader));
        output->write(reinterpret_cast<char const*>(dataProcessingHeader), sizeof(DataProcessingHeader));
        output->write(entry.payload, o2::framework::DataRefUtils::getPayloadSize(entry));
        LOG(INFO) << "wrote data, size " << o2::framework::DataRefUtils::getPayloadSize(entry);
      }
    });
  };

  std::vector<InputSpec> validBinaryInputs;
  auto onlyTimeframe = [](InputSpec const& input) {
    return input.lifetime == Lifetime::Timeframe;
  };

  auto noTimeframe = [](InputSpec const& input) {
    return input.lifetime != Lifetime::Timeframe;
  };

  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(validBinaryInputs), onlyTimeframe);
  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(unmatched), noTimeframe);

  DataProcessorSpec spec{
    "internal-dpl-global-binary-file-sink",
    validBinaryInputs,
    Outputs{},
    AlgorithmSpec(writerFunction),
    {{"outfile", VariantType::String, "dpl-out.bin", {"Name of the output file"}},
     {"keep", VariantType::String, "", {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION to save in outfile"}}}};

  return spec;
}

DataProcessorSpec CommonDataProcessors::getDummySink(std::vector<InputSpec> const& danglingOutputInputs)
{
  return DataProcessorSpec{
    "internal-dpl-dummy-sink",
    danglingOutputInputs,
    Outputs{},
  };
}

} // namespace framework
} // namespace o2
