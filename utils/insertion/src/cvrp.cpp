#include "head.h"
// #include<iostream>

Route* newroute(unsigned depotid){
	Node* head = new Node;
	head->value = depotid;
	head->length = 0.0;
	head->next = head;

	Route* route = new Route;
	route->head = head;
	route->length = 0.0;
	route->demand = 0;

	return route;
}

CVRPReturn* CVRPInsertion::randomInsertion(unsigned *order, float exploration = 1.0){
	// initialize ==============================
	// std::printf("initialization\n");
	unsigned cc = cvrpi->citycount;
	Node* vacant = nullptr;
	std::vector<Route*> routes;	
	unsigned capacity = cvrpi->capacity;
	{
		Node node;
		Node *lastnode = &node;
		for(unsigned i=0; i<cc; i++)
		{
			Node *thisnode = new Node;
			thisnode -> value = order[i];
			lastnode = lastnode->next = thisnode;
		}
		vacant = node.next;
		node.next = nullptr;
	}
	unsigned depot = cc;
	// vacant->printall(100);
	// start loop ==================================
	// std::printf("looping\n");
	for(unsigned i=0; i<cc; ++i){
		// get a city from vacant
		Node *curr;
		{
			curr = vacant;
			vacant = vacant->next, curr->next = nullptr;
		}
		unsigned currcity = curr -> value;
		float depotdist = cvrpi->getdistance(currcity, depot);
		float mincost = 2.0 * depotdist * exploration;
		unsigned currdemand = cvrpi->demand[currcity];
		Route* minroute = nullptr;
		Node* minnode = nullptr;
		// std::printf("find insert position for city %i\n", currcity);
		// get insert posiion with minimum cost
		for(std::vector<Route*>::iterator j = routes.begin(); j<routes.end(); ++j){
			Route* route = *j;
			if(route->demand + currdemand > capacity)
				continue;
			Node *headnode = route->head;
			Node *thisnode = headnode, *nextnode = headnode->next;
			float thisdist = mincost/2, nextdist = 0;
			do{
				nextnode = thisnode->next;
				nextdist = cvrpi->getdistance(nextnode->value, currcity);
				float delta = thisdist + nextdist - nextnode->length;
				if(delta < mincost)
					mincost = delta, minnode = thisnode, minroute = route;
				thisnode = nextnode, thisdist = nextdist;
			}while(nextnode!=headnode);
		}
		// std::printf("update status for city %i\n", currcity);
		// update state
		Route* route = nullptr;
		Node* pre = nullptr;
		if(minroute == nullptr){
			route = newroute(depot);
			pre = route -> head;
			pre->next->length = curr->length = depotdist;
			routes.push_back(route);
			mincost = depotdist * 2.0;
		}else{
			pre = minnode, route = minroute;
			curr->length = cvrpi->getdistance(pre->value, currcity);
			pre->next->length = cvrpi->getdistance(currcity, pre->next->value);
		}
		Node* next = pre->next;
		pre->next = curr, curr->next = next;
		route->demand += currdemand;
		route->length += mincost;
	}
	// std::printf("outputing\n");
	unsigned* norder = new unsigned[cc];
	unsigned len = routes.size()+1;
	unsigned* routesep = new unsigned[len];
	unsigned* routesepptr = routesep, accu=0;
	// get routes =========================
	while(!routes.empty()){
		Route* route = routes.back();
		routes.pop_back();
		Node* headnode = route->head;
		Node* currnode = headnode->next;
		*(routesepptr++) = accu;
		// std::printf("route: ");
		while(currnode!=headnode){
			// printf("%i ", currnode->value);
			norder[accu++] = currnode->value;
			currnode = currnode->next;
		}
		// std::printf(" cost: %f demand: %i\n", route->length, route->demand);
		// clean up
		Node *next = headnode->next;
		headnode->next = nullptr;
		delete next;
		delete route;
	}
	*routesepptr = accu;

	CVRPReturn *result = new CVRPReturn;
	result->order = norder;
	result->routes = len;
	result->routesep = routesep;

	return result;
}



