
## Notes on compilation

This module is compiled on x86-64 Linux with python 3.10 and numpy 1.23, please recomplie it before running if you are using a different setup.

The source code of numpy is required during compliation, please download it from https://github.com/numpy/numpy and include its path in makefile.

## Notes on insertion for CVRP

Our implementation of the insertion for CVRP differs from the random insertion for TSP in that the nodes are inserted in an azimuth-increasing order.
Another difference is that the former generates multiple routes under the capacity constraint. Consequently, the cost of creating a new route is considered in the insertion, and the quantity of generated routes is limited by the maximum vehicle number constraint [1].

---- 

[1] Hou, Q., Yang, J., Su, Y., Wang, X., & Deng, Y. Generalize Learned Heuristics to Solve Large-scale Vehicle Routing Problems in Real-time. In The Eleventh International Conference on Learning Representations.