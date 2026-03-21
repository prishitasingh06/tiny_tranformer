CXX = clang++
CXXFLAGS = -O2 -std=c++17

all: run

run: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o run

clean:
	rm -f run