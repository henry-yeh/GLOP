
#include <vector>
#include <math.h>

inline float calc_distance(float* a, float* b){
	float d1 = *a - *b, d2 = *(a + 1) - *(b + 1);
	return sqrtf32(d1*d1+d2*d2);
}

class Node
{
public:
    ~Node();
    Node *next = nullptr;
    unsigned value = 0;
    float length = 0;
    void printall(unsigned max);
    unsigned toList(Node **list, unsigned length);
};

class TSPinstance
{
public:
    unsigned citycount;
    float *citypos;

    TSPinstance(unsigned cc, float *cp)
    {
        citycount = cc;
        citypos = cp;
    };
    // ~TSPinstance();
    float getdist(unsigned a, unsigned b)
    {
        float *p1 = citypos + (a << 1), *p2 = citypos + (b << 1);
        float d1 = *p1 - *p2, d2 = *(p1 + 1) - *(p2 + 1);
        return sqrtf32(d1 * d1 + d2 * d2);
    };
};

class Insertion
{
public:
    Insertion(TSPinstance *tspi);
    // ~Insertion();
    unsigned *randomInsertion(unsigned *order);
    double getdistance()
    {
        return distance;
    };

private:
    TSPinstance *tspi;
    Node *vacant = nullptr;
    Node *route = nullptr;
    double distance;
    Node *getVacantNode();
    void initState(unsigned *order);
    void cleanup();
};

class CVRPInstance{
public:
	unsigned citycount;
	float *citypos;   // nx2
	unsigned *demand; // n
	float *depotpos;  // 2
	unsigned capacity;
	CVRPInstance(unsigned cc, float* cp, unsigned* dm, float* dp, unsigned cap){
		citycount = cc;
		citypos = cp;
		demand = dm;
		depotpos = dp;
		capacity = cap;
	}
	float getdistance(unsigned a, unsigned b){
		float* p1 = (a<citycount)?citypos + (a<<1):depotpos;
		float* p2 = (b<citycount)?citypos + (b<<1):depotpos;
		return calc_distance(p1, p2);
	}
};

struct CVRPReturn{
	unsigned routes;
	unsigned* order;
	unsigned* routesep;
};

class CVRPInsertion
{
public:
	CVRPInsertion(CVRPInstance* cvrpi);
	CVRPReturn *randomInsertion(unsigned *order, float exploration);

private:
	CVRPInstance* cvrpi;
};

struct Route{
	Node* head;
	unsigned demand;
	float length;
};
