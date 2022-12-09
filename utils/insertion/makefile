cxx = g++
cxxflags = -std=c++17 -O3 -fPIC -I/usr/include/python3.10 -I./

sources = $(wildcard src/*.cpp)
targets := $(patsubst %.cpp,%.o,$(sources))
depends := $(patsubst %.cpp,%.d,$(sources))

.PHONY: default clean test;

default: insertion.so;

-include $(depends)

$(targets): %.o: %.cpp
	$(cxx) $(cxxflags) -MMD -MP -c -o $@ $<

insertion.so: $(targets)
	$(cxx) $(cxxflags) --shared -o ./insertion.so -lpython3.10 $^

clean:
	rm -f src/*.o src/*.d *.o *.d *.so

test: insertion.so
	python3 -c "import insertion, numpy as np;order=np.arange(0,1000,dtype=np.uint32);print(insertion.random(np.random.rand(1000,2).astype(np.float32),order))" 
	

