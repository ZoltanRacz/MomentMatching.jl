### Relation with GMM
In the above example we have explained how the procedure works with SMM, but extending usage of our package routines to GMM is straightforward. With GMM usually one has a set of moment conditions that should hold with equality and rather than simulating data from a model, actual data are used to compute such conditions over the points in the parameter space. 

The user, therefore, in this case just needs to write their own code to compute such conditions and check how far away from zero they are. In other words, zero is the data moment to be used when computing the difference between model and data moments. 

Note that since the default version of `mdiff` scales by the average of a specific data moment (if one targets just a moment the mean is clearly the moment itself), if such value is zero, the user also needs to write a mode-specific `mdiff` function.