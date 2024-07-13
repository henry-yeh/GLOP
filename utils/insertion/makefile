cxx = g++
pythonversion = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
pythonroot = $(shell "python$(pythonversion)" -c "import sys; print(sys.prefix)")
numpyroot = $(shell "python$(pythonversion)" -c "import numpy,os; print(os.path.dirname(numpy.__file__))")
cxxflags = -std=c++17 -O3 -fPIC -I$(pythonroot)/include/python$(pythonversion) -I$(numpyroot)/core/include/

sources = $(wildcard src/*.cpp)
targets := $(patsubst %.cpp,%.o,$(sources))
depends := $(patsubst %.cpp,%.d,$(sources))

.PHONY: default clean testtsp testtsp_ne testcvrp;

default: insertion.so;

-include $(depends)

$(targets): %.o: %.cpp
	$(cxx) $(cxxflags) -MMD -MP -c -o $@ $<

insertion.so: $(targets)
	$(cxx) $(cxxflags) --shared -o ./insertion.so $^

clean:
	rm -f src/*.o src/*.d *.o *.d *.so

testtsp: insertion.so
	python3.8 -c "\
	import insertion, numpy as np;\
	order=np.arange(0,1000,dtype=np.uint32);\
	print(insertion.random(np.random.rand(1000,2).astype(np.float32),order,True))" 

testtsp_ne: insertion.so
	python3.8 -c "\
	import insertion,numpy as np;\
	order=np.arange(0,1000,dtype=np.uint32);\
	print(insertion.random(np.random.rand(1000,1000).astype(np.float32),order,False))" 


testcvrp: insertion.so
	python3 ./__init__.py