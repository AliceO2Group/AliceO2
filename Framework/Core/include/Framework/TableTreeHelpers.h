#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

// =============================================================================
namespace o2
{
namespace framework
{

// -----------------------------------------------------------------------------
// class branchIterator is a data iterator
// it basically allows to iterate over the elements of an arrow::table::Column
// and successively fill the branches with branchIterator::push()
// it is used in CommonDataProcessors::table2treeE
class branchIterator
{

  private:
  std::string brn;          // branch name
  arrow::ArrayVector chs;   // chunks
  Int_t nchs;               // number of chunks
  Int_t ich;                // chunk counter
  Int_t nr;                 // number of rows
  Int_t ir;                 // row counter

  // data buffers for each data type
  bool status = false;
  arrow::Type::type dt;
  std::string leaflist;

  TBranch *br   = nullptr;
  
  char *dbuf    = nullptr;
  void *v       = nullptr;
  
  Float_t  *vF  = nullptr;
  Double_t *vD  = nullptr;
  UShort_t *vs  = nullptr;
  UInt_t *vi    = nullptr;
  ULong64_t *vl = nullptr;
  Short_t *vS   = nullptr;
  Int_t *vI     = nullptr;
  Long64_t *vL  = nullptr;


  // initialize a branch
  bool branchini(TTree *tree)
  {
        
    // try to find branch in tree
    br = tree->GetBranch(brn.c_str());

    if (!br) {
      
      // create new branch of given data type
      switch (dt) {
        case arrow::Type::type::FLOAT:
          leaflist = brn+"/F";
          break;
        case arrow::Type::type::DOUBLE:
          leaflist = brn+"/D";
          break;
        case arrow::Type::type::UINT16:
          leaflist = brn+"/s";
          break;
        case arrow::Type::type::UINT32:
          leaflist = brn+"/i";
          break;
        case arrow::Type::type::UINT64:
          leaflist = brn+"/l";
          break;
        case arrow::Type::type::INT16:
          leaflist = brn+"/S";
          break;
        case arrow::Type::type::INT32:
          leaflist = brn+"/I";
          break;
        case arrow::Type::type::INT64:
          leaflist = brn+"/L";
          break;
        default:
          LOG(FATAL) << "Type not handled: " << dt << std::endl;
          break;          
      }

      br = tree->Branch(brn.c_str(),dbuf,leaflist.c_str());

    }
    if (!br)
      return false;
      
    return true;

  };
    
  
  // initialize chunk ib
  bool dbufini(Int_t ib)
  {
    // get next chunk of given data type dt
    switch (dt)
    {
      case arrow::Type::type::FLOAT:
        vF = (Float_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::FloatType>>(chs.at(ib))->raw_values();
        v = (void*) vF;
        break;
      case arrow::Type::type::DOUBLE:
        vD = (Double_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(chs.at(ib))->raw_values();
        v = (void*) vD;
        break;
      case arrow::Type::type::UINT16:
        vs = (UShort_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt16Type>>(chs.at(ib))->raw_values();
        v = (void*) vs;
        break;
      case arrow::Type::type::UINT32:
        vi = (UInt_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt32Type>>(chs.at(ib))->raw_values();
        v = (void*) vi;
        break;
      case arrow::Type::type::UINT64:
        vl = (ULong64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt64Type>>(chs.at(ib))->raw_values();
        v = (void*) vl;
        break;
      case arrow::Type::type::INT16:
        vS = (Short_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int16Type>>(chs.at(ib))->raw_values();
        v = (void*) vS;
        break;
      case arrow::Type::type::INT32:
        vI = (Int_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(chs.at(ib))->raw_values();
        v = (void*) vI;
        break;
      case arrow::Type::type::INT64:
        vL = (Long64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(chs.at(ib))->raw_values();
        v = (void*) vL;
        break;
      default:
        LOG(FATAL) << "Type not handled: " << dt << std::endl;
        break;          
      
    }
    br->SetAddress(v);
    
    // reset number of rows nr and row counter ir
    nr = chs.at(ib)->length();
    ir = 0;
    
    return true;
    
  };
    
  
  public:  

  branchIterator(TTree* tree, std::shared_ptr<arrow::Column> col)
  {
    brn  = col->name();
    dt   = col->type()->id();
    chs  = col->data()->chunks();
    nchs = chs.size();
    
    // initialize the branch
    status = branchini(tree);
    
    ich  = 0;
    status &= dbufini(ich);
    LOG(DEBUG) << "branchIterator: " << col->type()->ToString() << " with " << nchs << " chunks" << std::endl;

  };
  
  ~branchIterator()
  {
  };

  // has the iterator been properly initialized
  bool Status()
  {
    return status;
  }
    
  // fills buffer with next value
  // returns false if end of buffer reached
  bool push()
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
        return false;
      }
    } else {
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
        default:
          LOG(FATAL) << "Type not handled: " << dt << std::endl;
          break;
      }
    }
    br->SetAddress(v);
    ir++;
    
    return true;
    
  };
    
};

// -----------------------------------------------------------------------------
class TableToTree
{

  private:
  
    TTree* tr;

    // a list of branchIterator
    std::vector<branchIterator*> brits;
    
    // table to convert
    std::shared_ptr<arrow::Table> ta;

  
  public:
    
    TableToTree(std::shared_ptr<arrow::Table> table,
                TFile* file,
                const char* treename)
    {
    
      ta = table;
      
      // try to get the tree
      tr = (TTree*)file->Get(treename);
      
      // create the tree if it does not exist already
      if (!tr)
        tr = new TTree(treename,treename);
      
    }
    
    ~TableToTree()
    {
    
      // clean up branch iterators
      brits.clear();
    
    }
    
    bool AddBranch(std::shared_ptr<arrow::Column> col)
    {
      branchIterator *brit = new branchIterator(tr,col);
      if (brit->Status())
        brits.push_back(brit);
      
      return brit->Status();
      
    }

    bool AddAllBranches()
    {
      
      bool status = ta->num_columns()>0;
      for (auto ii=0; ii<ta->num_columns(); ii++)
      {
        branchIterator *brit =
          new branchIterator(tr,ta->column(ii));
        if (brit->Status())
        {
          brits.push_back(brit);
        } else {
          status = false;
        }
      }
      
      return status;
      
    }
    
    TTree* Process()
    {
    
      bool togo = true;
      while (togo)
      {
        // fill the tree
        tr->Fill();
  
        // update the branches
        for (auto ii=0; ii<ta->num_columns(); ii++)
          togo &= brits.at(ii)->push();

      }
      tr->Write();
      
      return tr;
    
    }

};


// -----------------------------------------------------------------------------
class columnIterator
{

  private:
  
    // all the possible TTreeReaderValue<T> types
    TTreeReaderValue<Float_t>   *vF = nullptr;
    TTreeReaderValue<Double_t>  *vD = nullptr;
    TTreeReaderValue<UShort_t>  *vs = nullptr;
    TTreeReaderValue<UInt_t>    *vi = nullptr;
    TTreeReaderValue<ULong64_t> *vl = nullptr;
    TTreeReaderValue<Short_t>   *vS = nullptr;
    TTreeReaderValue<Int_t>     *vI = nullptr;
    TTreeReaderValue<Long64_t>  *vL = nullptr;
  
    // all the possible arrow::TBuilder types
    arrow::FloatBuilder  *bF = nullptr;
    arrow::DoubleBuilder *bD = nullptr;
    arrow::UInt16Builder *bs = nullptr;
    arrow::UInt32Builder *bi = nullptr;
    arrow::UInt64Builder *bl = nullptr;
    arrow::Int16Builder  *bS = nullptr;
    arrow::Int32Builder  *bI = nullptr;
    arrow::Int64Builder  *bL = nullptr;
    
    bool status = false;
    EDataType dt;             // data type
    const char* cname;

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::shared_ptr<arrow::Field> field;
    std::shared_ptr<arrow::Array> ar;


  public:  

    columnIterator(TTreeReader* reader, const char* colname)
    {
  
      // find branch
      auto tree = reader->GetTree();
      if (!tree)
      {
        LOG(INFO) << "Can not locate tree!";
        return;
      }
      //tree->Print();
      auto br   = tree->GetBranch(colname);
      if (!br)
      {
        LOG(INFO) << "Can not locate branch " << colname;
        return;
      }
      cname = colname;

      TClass *cl;
      br->GetExpectedType(cl,dt);
      //LOG(INFO) << "Initialisation of TTreeReaderValue";
      //LOG(INFO) << "The column " << cname << " is of type " << dt;
      
      // initialize the TTreeReaderValue<T>
      //            the corresponding arrow::TBuilder
      //            the column schema
      // the TTreeReaderValue is incremented by reader->Next()
      // switch according to dt
      status = true;
      switch (dt)
      {
        case EDataType::kFloat_t:
          field = std::make_shared<arrow::Field>(cname,arrow::float32());
          vF = new TTreeReaderValue<Float_t>(*reader,cname);
          bF = new arrow::FloatBuilder(pool);
          break;
        case EDataType::kDouble_t:
          field = std::make_shared<arrow::Field>(cname,arrow::float64());
          vD = new TTreeReaderValue<Double_t>(*reader,cname);
          bD = new arrow::DoubleBuilder(pool);
          break;
        case EDataType::kUShort_t:
          field = std::make_shared<arrow::Field>(cname,arrow::uint16());
          vs = new TTreeReaderValue<UShort_t>(*reader,cname);
          bs = new arrow::UInt16Builder(pool);
          break;
        case EDataType::kUInt_t:
          field = std::make_shared<arrow::Field>(cname,arrow::uint32());
          vi = new TTreeReaderValue<UInt_t>(*reader,cname);
          bi = new arrow::UInt32Builder(pool);
          break;
        case EDataType::kULong64_t:
          field = std::make_shared<arrow::Field>(cname,arrow::uint64());
          vl = new TTreeReaderValue<ULong64_t>(*reader,cname);
          bl = new arrow::UInt64Builder(pool);
          break;
        case EDataType::kShort_t:
          field = std::make_shared<arrow::Field>(cname,arrow::int16());
          vS = new TTreeReaderValue<Short_t>(*reader,cname);
          bS = new arrow::Int16Builder(pool);
          break;
        case EDataType::kInt_t:
          field = std::make_shared<arrow::Field>(cname,arrow::int32());
          vI = new TTreeReaderValue<Int_t>(*reader,cname);
          bI = new arrow::Int32Builder(pool);
          break;
        case EDataType::kLong64_t:
          field = std::make_shared<arrow::Field>(cname,arrow::int64());
          vL = new TTreeReaderValue<Long64_t>(*reader,cname);
          bL = new arrow::Int64Builder(pool);
          break;
        default:
          LOG(FATAL) << "Type not handled: " << dt << std::endl;
          break;          
      }

    }
    
    ~columnIterator()
    {
      
      // delete all pointers
      if (vF) delete vF;
      if (vD) delete vD;
      if (vs) delete vs;
      if (vi) delete vi;
      if (vl) delete vl;
      if (vS) delete vS;
      if (vI) delete vI;
      if (vL) delete vL;

      if (bF) delete bF;
      if (bD) delete bD;
      if (bs) delete bs;
      if (bi) delete bi;
      if (bl) delete bl;
      if (bS) delete bS;
      if (bI) delete bI;
      if (bL) delete bL;

    };

    // has the iterator been properly initialized
    bool Status()
    {
      return status;
    }
    
    // copy the TTreeReaderValue to the arrow::TBuilder
    void* push()
    {
      
      // switch according to dt
      switch (dt)
      {
        case EDataType::kFloat_t:
          bF->Append(**vF);
          break;
        case EDataType::kDouble_t:
          bD->Append(*vD->Get());
          break;
        case EDataType::kUShort_t:
          bs->Append(*vs->Get());
          break;
        case EDataType::kUInt_t:
          bi->Append(*vi->Get());
          break;
        case EDataType::kULong64_t:
          bl->Append(*vl->Get());
          break;
        case EDataType::kShort_t:
          bS->Append(*vS->Get());
          break;
        case EDataType::kInt_t:
          bI->Append(*vI->Get());
          break;
        case EDataType::kLong64_t:
          bL->Append(*vL->Get());
          break;
        default:
          LOG(FATAL) << "Type not handled: " << dt << std::endl;
          break;          
      }
    }
    
    std::shared_ptr<arrow::Array> Array()
    {
      return ar;
    }
    
    std::shared_ptr<arrow::Field> Schema()
    {
      return field;
    }
    
    // finish the arrow::TBuilder
    // with this ar is prepared to be used in arrow::Table::Make
    void* finish()
    {
    
      //LOG(INFO) << "columnIterator::finish " << dt;

      // switch according to dt
      switch (dt)
      {
        case EDataType::kFloat_t:
          bF->Finish(&ar);
          break;
        case EDataType::kDouble_t:
          bD->Finish(&ar);
          break;
        case EDataType::kUShort_t:
          bs->Finish(&ar);
          break;
        case EDataType::kUInt_t:
          bi->Finish(&ar);
          break;
        case EDataType::kULong64_t:
          bl->Finish(&ar);
          break;
        case EDataType::kShort_t:
          bS->Finish(&ar);
          break;
        case EDataType::kInt_t:
          bI->Finish(&ar);
          break;
        case EDataType::kLong64_t:
          bL->Finish(&ar);
          break;
        default:
          LOG(FATAL) << "Type not handled: " << dt << std::endl;
          break;          
      }
    }
  
};

// -----------------------------------------------------------------------------
class TreeToTable
{

  private:
  
    // the TTreeReader allows to efficiently loop over
    // the rows of a TTree
    TTreeReader *reader;

    // a list of columnIterator*
    std::vector<std::shared_ptr<columnIterator>> colits;

    // Append next set of branch values to the
    // corresponding table columns
    void* push()
    {
      for (auto colit : colits)
        colit.get()->push();
    }
    

  public:  

    TreeToTable(TTree *tree)
    {
      // initialize the TTreeReader
      //LOG(INFO) << "Initialisation of TTreeReader";
      reader = new TTreeReader(tree);
    }
    
    ~TreeToTable()
    {
      if (reader) delete reader;
    };
    
    // add a column to be included in the arrow::table
    bool AddColumn(const char* colname)
    {
      auto colit = std::make_shared<columnIterator>(reader,colname);
      auto stat = colit.get()->Status();
      if (stat)
        colits.push_back(std::move(colit));
      
      return stat;
    }
    
    // add all columns
    bool AddAllColumns()
    {
      // get a list of column names
      auto tree = reader->GetTree();
      if (!tree)
      {
        LOG(INFO) << "Tree not found!";
        return false;
      }
      auto branchList = tree->GetListOfBranches();
      
      // loop over branches
      bool status = !branchList->IsEmpty();
      for (Int_t ii=0; ii<branchList->GetEntries(); ii++)
      {
        auto br = (TBranch*)branchList->At(ii);
        
        // IMPROVE: make sure that a column is not added more than one time
        auto colit = std::make_shared<columnIterator>(reader,br->GetName());
        if (colit.get()->Status())
        {
          colits.push_back(std::move(colit));
        } else {
          status = false;
        }
      }
      //LOG(INFO) << "Status " << status; 
      //LOG(INFO) << "Number of columns " << colits.size(); 
      
      return status;
    }
    
    int num_columns()
    {
      return colits.size();
    }
    
    // do the looping with the TTreeReader and fill the table
    std::shared_ptr<arrow::Table> Process()
    {
      //LOG(INFO) << "TreeToTable::Processe";
      //LOG(INFO) << "There are " << reader->GetEntries() << " entries.";
      
      // copy all values from the tree to the table builders
      reader->Restart();
      while (reader->Next()) push();

      // prepare the elements needed to create the final table
      std::vector<std::shared_ptr<arrow::Array>> array_vector; 
      std::vector<std::shared_ptr<arrow::Field>> schema_vector; 
      for (auto colit : colits)
      {
        colit.get()->finish();
        array_vector.push_back(colit.get()->Array());
        schema_vector.push_back(colit.get()->Schema());
      }
      auto fields = std::make_shared<arrow::Schema>(schema_vector);

      // create the final table
      auto ta = (arrow::Table::Make(fields,array_vector));
      //LOG(INFO) << "Created table: " << ta.get()->Validate();
      //LOG(INFO) << "  fields:  " << ta.get()->schema()->ToString();
      //LOG(INFO) << "  columns: " << ta.get()->num_columns();
      //LOG(INFO) << "  rows:    " << ta.get()->num_rows();

      return ta;

    }

};


// -----------------------------------------------------------------------------
} // namespace framework
} // namespace o2

// =============================================================================
