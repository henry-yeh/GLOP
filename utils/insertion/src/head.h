
#define THREADS 16
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>

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