#include "head.h"
#include <iostream>

Insertion::Insertion(TSPinstance *tspinstance)
{
    tspi = tspinstance;
}

void Insertion::initState(unsigned *order)
{
    cleanup();
    unsigned cc = tspi->citycount;
    Node node;
    Node *lastnode = &node;
    for (unsigned i = 0; i < cc; i++)
    {
        Node *thisnode = new Node;
        thisnode->value = order[i];
        lastnode = lastnode->next = thisnode;
    }
    vacant = node.next;
    node.next = nullptr;
    // vacant->printall(cc + 10);
}

void Insertion::cleanup()
{
    if (vacant != nullptr)
        delete vacant;
    if (route != nullptr)
        delete route;
    vacant = nullptr;
    route = nullptr;
}

Node *Insertion::getVacantNode()
{
    Node *result = vacant;
    if (vacant != nullptr)
    {
        vacant = vacant->next;
        result->next = nullptr;
    }
    return result;
}

unsigned *Insertion::randomInsertion(unsigned *order)
{
    initState(order);
    unsigned cc = tspi->citycount;
    // generate initial route with 2 nodes
    {
        Node *node1 = getVacantNode();
        Node *node2 = getVacantNode();
        node2->next = node1;
        node1->next = node2;
        route = node1;
        node1->length = node2->length = tspi->getdist(node1->value, node2->value);
    }

    for (unsigned i = 2; i < cc; i++)
    {
        // get a city from vacant
        Node *curr = getVacantNode();
        unsigned city = curr->value;
        // unsigned routelen = i - 1;

        // get target list and distances
        // and get insert position with minimum cost
        Node *thisnode = route, *nextnode = thisnode->next;
        float thisdist = tspi->getdist(thisnode->value, city), nextdist = 0;
        Node *minnode = thisnode;
        float mindelta = INFINITY;
        float td = 0.0, nd = 0.0;

        for (unsigned j = 0; j < i; j++)
        {
            nextnode = thisnode->next;
            nextdist = tspi->getdist(nextnode->value, city);
            float delta = thisdist + nextdist - nextnode->length;
            if (delta < mindelta)
            {
                mindelta = delta, minnode = thisnode;
                td = thisdist, nd = nextdist;
            }
            thisnode = nextnode, thisdist = nextdist;
        }

        // insert the selected node
        Node *pre = minnode, *next = minnode->next;
        pre->next = curr, curr->next = next;
        curr->length = td, next->length = nd;
    }
    // get node order
    Node *node = route;
    distance = 0.0;
    unsigned *output = new unsigned[cc];
    for (unsigned i = 0; i < cc; i++)
    {
        output[i] = node->value;
        distance += node->length;
        node = node->next;
    }

    // cleanup();
    return output;
}

unsigned Node::toList(Node **list, unsigned length)
{
    /* returns actual length */
    unsigned i;
    Node *curr = this;
    for (i = 0; i < length && curr != nullptr; i++)
    {
        list[i] = curr;
        curr = curr->next;
    }
    return i;
}

void Node::printall(unsigned max)
{
    Node *thisnode = this;
    for (unsigned i = 0; i < max && thisnode != nullptr; i++)
    {
        // prevent loops
        std::cout << thisnode->value << ' ';
        thisnode = thisnode->next;
    }
    std::cout << std::endl;
}

Node::~Node()
{
    if (next != nullptr)
        delete next;
}