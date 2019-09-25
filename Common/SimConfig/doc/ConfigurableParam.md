\page refCommonSimConfig SimConfig

# Configurable Parameters

This is a short README for configurable parameters as offered by 
the ConfigurableParameter class.

# Introduction

The ConfigurableParameter class implements the demand
* to have simple variables (under a compound namespace) be declared as 'knobs'
of an algorithm in order to be able to change/configure their value without recompilation.
* to have such declared variables be automatically registered in a parameter database or manager instance.
* to be able to configure/change the parameter values in a textual way (for instance from the command line)
* to provide automatic serialization / deserialization techniques (to text files or binary blobs) -- for instance to load from CCDB or to pass along parameter snapshots to other processing stages.
* to keep track of who changes parameters (provenance tracking). 

# Example / HowTo:

Imagine some algorithms `algorithmA` depends on 2 parameters `p1` and `p2` which you want to be able to configure.

You would do the following steps:
  1. Declare a parameter class listing the parameters and their default values.
     ```c++
     struct ParamA : ConfigurableParamHelper<ParamA> {
       int p1 = 10;
       double p2 = 1.23;
       // boilerplate stuff + make parameters known under key "A"
       O2ParamDef(ParamA, "A");
     };
     ```
  2. Put 
     ```c++
     O2ParamImpl(ParamA);
     ```
     in some source file, to generate necessay symbols needed for linking.
  3. Access and use the parameters in the code.
     ```c++
     void algorithmA() {
       // get the parameter singleton object
       auto& pa = ParamA::Instance();
       // access the variables in your code
       doSomething(pa.p1, pa.p2);
     }
     ```
    
Thereafter, the parameter `ParamA` is automatically registered in a parameter registry and can be read/influenced/serialized through this. The main influencing functions are implemented as static functions on the `ConfigurableParam` class. For example, the following things will be possible:
* get a value by string key, addressing a specific parameter:
  ```c++
   auto p = ConfigurableParam::getValue<double>("A.p2");
  ```
  where the string keys are composed of a primary key and the variable name of the parameter.
* set/modify a value by a key string
  ```c++
  ConfigurableParam::setValue<double>("A.p2", 10);
  ```
  where the string keys are composed of a primary key and the variable name of the parameter.
  Side note: This API allows to influence values from a string, for instance obtained from command line:
  ```c++
  ConfigurableParam::fromString("A.p2=10,OtherParam.a=-1.");
  ```
  The system will complain if a non-existing string key is used.
  
* serialize the configuration to a ROOT snapshot or to formats such as JSON or INI
  ```c++
  ConfigurableParam::toINI(filename);
  ConfigurableParam::toJSON(filename); // JSON repr. of param registry
  ConfigurableParam::toCCDB(filename); // CCDB snapshot of param registry
  ```
* retrieve parameters from CCDB snapshot
  ```c++
  ConfigurableParam::fromCCDB(filename);
  ```
* **Provenance tracking**: The system can keep track of the origin of values. Typically, few stages of modification can be done:
  - default initialization of parameters from code (CODE)
  - initialization/overwritting from a (CCDB) snapshot file
  - overriding by user during runtime (RT)

  The registry can keep track of who supplied the current (decisive) value for each parameter: CODE, CCDB or RT. If a parameter is not changed it keeps the state of the previous stage.

# Parameter modifcation from command line and ini file

As mentioned above, it is possible to overwrite configuration parameters from the command line. This can be done using any of the DPL exacutables, e.g. `o2-sim`, `o2-sim-digitizer-workflow`, `o2-tpc-reco-workflow`, ...

## Command line option
In order to modify a parameter from the command line, the syntax `Key.param=value` has to be used. **NOTE:** there must be no spaced before and after the `=`! Multiple key value pairs are separated by a `;`:
```
o2-sim --configKeyValues 'A.p1=1;A.p2=2.56'
```
Not all parameters need to be defined. Non-defined parameters will use the default value defined in the parameter struct.

## ini file option
The layout of the ini-file is the following:
```EditorConfig
[Key1]
param1=value1
param2=value2
[Key2]
param1=value1
```

consider e.g. a file `paramA.ini`:
```EditorConfig
[A]
p1=1
```

This can be called from the command line using
```
o2-sim-digitizer-workflow --configFile paramA.ini
```
Again, not all parameters need to be defined.

# Further technical details

* Parameter classes are **read-only** singleton classes/structs. As long as the user follows the pattern to inherit from `ConfigurableParamHelper<T>` and to use the macro `O2ParamDef()` everything is implemented automatically.
* We use the `ROOT C++` introspection facility to map the class layout to a textual configuration. Hence ROOT dictionaries are needed for parameter classes.
* BOOST property trees are used for the mapping to JSON/INI files but this is an internal detail and might change in future.

# Limitations

* Parameter classes may only contain simple members! Currently the following types are supported
    * simple pods (for example `double x; char y;`)
    * std::string
    * fixed size arrays of pods using the ROOT way to serialize:
       ```c++
       static constexpr int N=3; //!
       double array[N] = {1, 2, 3}; //[N] -- note the [N] after //!!
       ```
    * array parameters need to be addressed textual with an index operator:
      ```c++
      ConfigurableParam::fromString("ParamA.array[2]=10");
      ```
      It is planned to offer more flexible ways to set arrays (see below).
    * there is currently no support for pointer types or objects. Parameter classes may not be nested.
* At the moment serialization works on the whole paramter registry (on the whole set of parameters). Everything is written to the same file or snapshot.

# Wish list / planned improvements

* Offer a more flexible way to set array types (to represent them in strings), for example
  ```c++
  ConfigurableParam::fromString("ParamA.array = {5, 6, 7}");
  ```
* Offer a better more flexible way to read from CCDB.
  * read from complete snapshots (give single file name) **or** read from different files reciding in paths corresponding to key/namespace.
* Of a more flexible way to serialize, for example allowing to output configuration to different files.
* Be able to change the configuration from a text file (next to current possibility from command line)
* support for few important stl containers (std::array, std::vector)
* a better way to define stages (CODE, CCDB, RT) and to transition between them; potentially more allowed stages
* take away more boilerplate: Automatic creation of dictionaries for parameter classes.
