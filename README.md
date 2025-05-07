
***Functionality***: Utilized a threadpool initialized with 2*num_cores threads that received tasks from build_tree() and knn_search() functions

build_tree: parallelized only till a certain threshold (number of points) since overhead for queueing the job and waiting for it was not worth, rest is done serially

knn_search: parallelized by enqueueing each node search and returning the neighbors + writing to the results file 

Compile with: ``` make all ```

Create training data: ```./training_data n_points d_dimension distribution ```

Create query points: ```./query_points n_points d_dimension number_of_neighbors```

K-nn search: ```./k-nn n_cores training_file query_file result_file```

*d_dimension of training and query must match
