# Estimation

## Model setup
The first step of the estimation routine is for the user to define her own model type through:
```@docs
EstimationSetup
```
For instance, running:
```

```
defines an EstimationSetup type with mode called Model1, modelname called benchmark and typemom set1.

Once EstimationSetup has been defined, 

```@docs
AuxiliaryParameters
```
```@docs
PredrawnShocks
```
```@docs
PreallocatedContainers
```
%```@docs
%NumParMM
%```
```@docs
ParMM
```
## Numerical routines
The main function to run the estimation routine is: 

```@docs
estimation
```

In addition to the model setup elements described above, the user can also define her preferred computational settings:

```@docs
ComputationSettings
```

Furthermore, the user can decide:
- whether to save the final estimation results through `saving`;
- whether to save the evaluated moments for the best (i.e., those returning the lowest objective function values) `number_bestmodel` points through `saving_bestmodel`;
- whether to add a specific suffix to saved files via `filename_suffix`;
- whether to proceed with the estimation routine until the end if evaluation of the objective function at a specific candidate point returns an error `errorcatching`.

Additional model-specific arguments needed for model initialization can be passed through via `vararg...`.

Results of the estimation routine are stored in a dedicated structure:
```@docs
EstimationResult
```

### Global stage

### Local stage

### Comparison with other algorithms

## Multithreading and multiprocessing

### Locally

### Cluster

## Two-step estimation



