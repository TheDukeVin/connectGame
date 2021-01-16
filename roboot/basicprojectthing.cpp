
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <algorithm>
#define numlayers 3
#define maxNodes 10
#define mutationRate 0.01
#define numNets 689
#define top 100
using namespace std;

ofstream fout ("basicproject.out");
ifstream fin ("basicproject.in");


struct network{
	int numNodes[numlayers+1];
	double weights[numlayers][maxNodes][maxNodes];
	double bias[numlayers][maxNodes];
	double activation[numlayers+1][maxNodes];
	double error;
	
	void randomize();
	void initialize();
	void copy(network net);
	void mutate();
	void pass();
	void printActs();
	void storeNet();
	void readNet();
};

double squ(double x){
	return x*x;
}

double nonlinear(double x){
	return tanh(x);
}

void network::randomize(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] = ((double)rand() / RAND_MAX)*2-1;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] = ((double)rand() / RAND_MAX)*2-1;
		}
	}
}

void network::initialize(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] = 0;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] = 0;
		}
	}
}

void network::copy(network net){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] = net.weights[l][i][j];
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] = net.bias[l][i];
		}
	}
}

void network::mutate(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] += (((double)rand() / RAND_MAX)*2-1) * mutationRate;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] = (((double)rand() / RAND_MAX)*2-1) * mutationRate;
		}
	}
}

void network::pass(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			activation[l+1][i] = bias[l][i];
			for(j=0; j<numNodes[l]; j++){
				activation[l+1][i] += weights[l][j][i] * activation[l][j];
			}
			activation[l+1][i] = nonlinear(activation[l+1][i]);
		}
	}
}

void network::storeNet(){
	int l,i,j;
	for(l=0; l<=numlayers; l++){
		fout<<numNodes[l]<<' ';
	}
	fout<<'\n';
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				fout<<weights[l][i][j]<<' ';
			}
			fout<<'\n';
		}
		fout<<'\n';
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			fout<<bias[l][i]<<' ';
		}
		fout<<'\n';
	}
}

void network::readNet(){
	int l,i,j;
	for(l=0; l<=numlayers; l++){
		fin>>numNodes[l];
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				fin>>weights[l][i][j];
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			fin>>bias[l][i];
		}
	}
}

void network::printActs(){
	int l,i;
	for(l=0; l<=numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			cout<<activation[l][i]<<' ';
		}
		cout<<'\n';
	}
}

struct testData{
	double nums[3];
	double e;
};

testData data[8];

struct population{
	network nets[numNets];
	
	void randomizeNets();
	void test();
	void selection();
};

int numNodes[4] = {3,4,4,1};

void population::randomizeNets(){
	int n,i;
	for(n=0; n<numNets; n++){
		for(i=0; i<=numlayers; i++){
			nets[n].numNodes[i] = numNodes[i];
		}
		nets[n].randomize();
	}
}

void population::test(){
	int n,t,i;
	for(n=0; n<numNets; n++){
		nets[n].error = 0;
		for(t=0; t<8; t++){
			for(i=0; i<3; i++){
				nets[n].activation[0][i] = data[t].nums[i];
			}
			nets[n].pass();
			nets[n].error += squ(nets[n].activation[numlayers][0] - data[t].e);
		}
	}
}

bool operator < (network n1, network n2){
	return n1.error < n2.error;
}

void population::selection(){
	sort(nets,nets+numNets);
	int i;
	for(i=top; i<numNets; i++){
		nets[i].copy(nets[i%top]);
		nets[i].mutate();
	}
}

int main(){
	srand (time(NULL));
	
	int i,j,k;
	for(i=0; i<2; i++){
		for(j=0; j<2; j++){
			for(k=0; k<2; k++){
				data[i*4+j*2+k].nums[0] = i*2-1;
				data[i*4+j*2+k].nums[1] = j*2-1;
				data[i*4+j*2+k].nums[2] = k*2-1;
				//data[i*4+j*2+k].e = max(min((i+j+k)*2-3,1),-1);
				//data[i*4+j*2+k].e = 1;
				//data[i*4+j*2+k].e = ((i+j+k)%2)*2-1;
				//data[i*4+j*2+k].e = (((i+j+k)%2)*2-1)*0.3;
				//data[i*4+j*2+k].e = (i+1.0)*(j+1)*(k+1) / 8;
			}
		}
	}
	data[0].e = -1;
	data[1].e = 1;
	data[2].e = -1;
	data[3].e = -1;
	data[4].e = 1;
	data[5].e = -1;
	data[6].e = -1;
	data[7].e = 1;
	
	population pop;
	pop.randomizeNets();
	for(i=0; i<1000; i++){
		pop.test();
		pop.selection();
		if(i%100 == 0){
			cout<<pop.nets[0].error<<'\n';
		}
	}
	cout<<pop.nets[0].error<<'\n';
	sort(pop.nets,pop.nets);
	
	network veryneat;
	for(i=0; i<=numlayers; i++){
		veryneat.numNodes[i] = numNodes[i];
	}
	veryneat.copy(pop.nets[0]);
	
	veryneat.storeNet();
	for(i=0; i<8; i++){
		for(j=0; j<3; j++){
			veryneat.activation[0][j] = data[i].nums[j];
		}
		veryneat.pass();
		cout<<"ANSWER: "<<data[i].e<<" Network: "<<veryneat.activation[numlayers][0]<<'\n';
	}
	
	return 0;
}
