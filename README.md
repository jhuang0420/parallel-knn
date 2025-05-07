[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MunaVo0t)

***Notes to grader***: Utilized a threadpool initialized with 2*num_cores threads that received tasks from my build_tree and knn_search functions
Results seemed accurate (when tested with low dimensionality), hopefully also works with higher 

build_tree: parallelized only till a certain threshold (number of points) since overhead for queueing the job and waiting for it was not worth, rest is done serially

knn_search: parallelized by enqueueing each node search and returning the neighbors + writing to the results file 
