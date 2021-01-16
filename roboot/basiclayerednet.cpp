
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#define numlayers 3
#define maxNodes 10
#define learnRate 0.0001
using namespace std;

ofstream fout ("test2.out");
ifstream fin ("test2.in");

struct network{
	int numNodes[numlayers+1];
	double weights[numlayers][maxNodes][maxNodes];
	double bias[numlayers][maxNodes];
	double activation[numlayers+1][maxNodes];
	double intermediate[numlayers][maxNodes];
	double Dbias[numlayers][maxNodes];
	double Dweights[numlayers][maxNodes][maxNodes];
	double Dactivation[numlayers+1][maxNodes];
	double Dintermediate[numlayers][maxNodes];
	double sumDbias[numlayers][maxNodes];
	double sumDweights[numlayers][maxNodes][maxNodes];
	
	void randomize();
	void initialize();
	void pass();
	void printActs();
	void storeNet();
	void readNet();
	void backProp();
	void initSum();
	void updateSum();
	void updateNet();
	void descent();
	void train(int iterations);
};

double squ(double x){
	x*x;
}

double nonlinear(double x){
	return tanh(x);
}

double dnonlinear(double x){
	return 1-squ(tanh(x));
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

void network::pass(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			intermediate[l][i] = bias[l][i];
			for(j=0; j<numNodes[l]; j++){
				intermediate[l][i] += weights[l][j][i] * activation[l][j];
			}
			activation[l+1][i] = nonlinear(intermediate[l][i]);
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

double expected[maxNodes];

void network::backProp(){
	int l,i,j;
	for(i=0; i<numNodes[numlayers]; i++){
		Dactivation[numlayers][i] = 2*(activation[numlayers][i] - expected[i]);
	}
	double sum;
	for(l=numlayers-1; l>=0; l--){
		for(i=0; i<numNodes[l+1]; i++){
			Dintermediate[l][i] = Dactivation[l+1][i] * dnonlinear(intermediate[l][i]);
		}
		for(i=0; i<numNodes[l]; i++){
			sum = 0;
			for(j=0; j<numNodes[l+1]; j++){
				sum += Dintermediate[l][j] * weights[l][i][j];
			}
			Dactivation[l][i] = sum;
		}
		for(i=0; i<numNodes[l+1]; i++){
			Dbias[l][i] = Dintermediate[l][i];
		}
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				Dweights[l][i][j] = Dintermediate[l][j] * activation[l][i];
			}
		}
	}
}

void network::initSum(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				sumDweights[l][i][j] = 0;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			sumDbias[l][i] = 0;
		}
	}
}

void network::updateSum(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				sumDweights[l][i][j] += Dweights[l][i][j];
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			sumDbias[l][i] += Dbias[l][i];
		}
	}
}

void network::updateNet(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] -= sumDweights[l][i][j] * learnRate;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] -= sumDbias[l][i] * learnRate;
		}
	}
}

/*
//test 1
network yosup;
int main(){
	yosup.readNet();
	yosup.activation[0][0] = 1;
	yosup.activation[0][1] = 2;
	yosup.activation[0][2] = 3;
	yosup.pass();
	yosup.printActs();
	yosup.storeNet();
}
*/

struct testData{
	double nums[3];
	double e;
};

testData data[8];

void network::descent(){
	int t,i;
	initSum();
	for(t=0; t<8; t++){
		for(i=0; i<3; i++){
			activation[0][i] = data[t].nums[i];
		}
		pass();
		expected[0] = data[t].e;
		backProp();
		updateSum();
	}
	updateNet();
}

void network::train(int iterations){
	int i;
	for(i=0; i<iterations; i++){
		descent();
	}
}

network veryneat;
int main(){
	srand (time(NULL));
	veryneat.numNodes[0] = 3;
	veryneat.numNodes[1] = 4;
	veryneat.numNodes[2] = 4;
	veryneat.numNodes[3] = 1;
	veryneat.randomize();
	
	int i,j,k;
	for(i=0; i<2; i++){
		for(j=0; j<2; j++){
			for(k=0; k<2; k++){
				data[i*4+j*2+k].nums[0] = i*2-1;
				data[i*4+j*2+k].nums[1] = j*2-1;
				data[i*4+j*2+k].nums[2] = k*2-1;
				data[i*4+j*2+k].e = max(min((i+j+k)*2-3,1),-1);
				//data[i*4+j*2+k].e = 1;
				//data[i*4+j*2+k].e = ((i+j+k)%2)*2-1;
			}
		}
	}
	
	/*
	veryneat.activation[0][0] = -1;
	veryneat.activation[0][1] = -1;
	veryneat.activation[0][2] = -1;
	
	veryneat.pass();
	
	expected
	
	veryneat.backProp();
	*/
	
	
	
	veryneat.train(1000);
	
	veryneat.storeNet();
	for(i=0; i<8; i++){
		for(j=0; j<3; j++){
			veryneat.activation[0][j] = data[i].nums[j];
		}
		veryneat.pass();
		cout<<"ANSWER: "<<data[i].e<<" Network: "<<veryneat.activation[2][0]<<'\n';
	}
	
}
