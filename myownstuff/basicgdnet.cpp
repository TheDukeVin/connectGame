
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#define numlayers 3
#define maxNodes 10
#define learnRate 0.01
#define numSamples 45
#define sampleInp 3
using namespace std;

ofstream fout ("bgd.out");
ifstream fin ("bgd.in");

struct sample{
	double nums[3];
	double e;
};

sample makeSample(double x1, double x2, double x3, double e){
  sample s;
  s.nums[0] = x1;
  s.nums[1] = x2;
  s.nums[2] = x3;
  s.e = e;
  return s;
}

sample testData[numSamples];

struct network{
	int numNodes[numlayers+1];
	double weights[numlayers][maxNodes][maxNodes];
	double bias[numlayers][maxNodes];
	double activation[numlayers+1][maxNodes];
  double inter[numlayers][maxNodes];

  double Dbias[numlayers][maxNodes];
  double Dweights[numlayers][maxNodes][maxNodes];
  double Dactivation[numlayers+1][maxNodes];

  double Sbias[numlayers][maxNodes];
  double Sweights[numlayers][maxNodes][maxNodes];
	
	void randomize();
	void initialize();

	void pass();
  double cost();
  void backProp();
  void delComp();
  void printD();
  void initS();
  void increment();
  void learn();

  void trainingCycle();
  void trainingInfo();
  double totalCost();

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

double dnonlinear(double x){
  return 1-squ(tanh(x));
}

double randVal(){
  return ((double)rand() / RAND_MAX)*2-1;
}

void network::randomize(){
	int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] = randVal()*0.1;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] = randVal()*0.1;
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
			inter[l][i] = bias[l][i];
			for(j=0; j<numNodes[l]; j++){
				inter[l][i] += weights[l][j][i] * activation[l][j];
			}
			activation[l+1][i] = nonlinear(inter[l][i]);
		}
	}
}

double expected[3];

double network::cost(){
  pass();
  int i;
  double sum = 0;
  for(i=0; i<numNodes[numlayers]; i++){
    sum += squ(activation[numlayers][i]-expected[i]);
  }
  return sum;
}

void network::backProp(){
  pass();
  int l,i,j;
  for(i=0; i<numNodes[numlayers]; i++){
    Dactivation[numlayers][i] = 2 * (activation[numlayers][i] - expected[i]);
  }
  for(l=numlayers-1; l>=0; l--){
    for(i=0; i<numNodes[l+1]; i++){
      Dbias[l][i] = Dactivation[l+1][i] * dnonlinear(inter[l][i]);
      for(j=0; j<numNodes[l]; j++){
        Dweights[l][j][i] = Dbias[l][i] * activation[l][j];
      }
    }
    for(i=0; i<numNodes[l]; i++){
      Dactivation[l][i] = 0;
      for(j=0; j<numNodes[l+1]; j++){
        Dactivation[l][i] += Dbias[l][j] * weights[l][i][j];
      }
    }
  }
}

void network::delComp(){
  int l,i,j;
  double initCost = cost();
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] += 0.01;
        Dweights[l][i][j] = (cost()-initCost)/0.01;
        weights[l][i][j] -= 0.01;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] += 0.01;
      Dbias[l][i] = (cost()-initCost)/0.01;
      bias[l][i] -= 0.01;
		}
	}
}

void network::printD(){
  int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				cout<<Dweights[l][i][j]<<' ';
			}
      cout<<'\n';
		}
    cout<<'\n';
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			cout<<Dbias[l][i]<<' ';
		}
    cout<<'\n';
	}
}

void network::initS(){
  int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				Sweights[l][i][j] = 0;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			Sbias[l][i] = 0;
		}
	}
}

void network::increment(){
  int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				Sweights[l][i][j] += Dweights[l][i][j];
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			Sbias[l][i] += Dbias[l][i];
		}
	}
}

void network::learn(){
  int l,i,j;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] -= Sweights[l][i][j] * learnRate;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] -= Sbias[l][i] * learnRate;
		}
	}
}

/*
void network::learn(){
  int l,i,j;
  double sum = 0;
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
        sum += squ(Sweights[l][i][j]);
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
      sum += squ(Sbias[l][i]);
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l]; i++){
			for(j=0; j<numNodes[l+1]; j++){
				weights[l][i][j] -= Sweights[l][i][j]/sqrt(sum) * learnRate;
			}
		}
	}
	for(l=0; l<numlayers; l++){
		for(i=0; i<numNodes[l+1]; i++){
			bias[l][i] -= Sbias[l][i]/sqrt(sum) * learnRate;
		}
	}
}
*/

void network::trainingCycle(){
  int i,j;
  initS();
  for(i=0; i<numSamples; i++){
    for(j=0; j<sampleInp; j++){
      activation[0][j] = testData[i].nums[j];
    }
    expected[0] = testData[i].e;
    backProp();
    increment();
  }
  learn();
}

void network::trainingInfo(){
  int i,j;
  for(i=0; i<numSamples; i++){
    cout<<"Data: ";
    for(j=0; j<sampleInp; j++){
      activation[0][j] = testData[i].nums[j];
      cout<<testData[i].nums[j]<<' ';
    }
    pass();
    cout<<"Expected: "<<testData[i].e<<"Network: "<<activation[numlayers][0]<<'\n';
  }
}

double network::totalCost(){
  int i,j;
  double sum = 0;
  for(i=0; i<numSamples; i++){
    for(j=0; j<sampleInp; j++){
      activation[0][j] = testData[i].nums[j];
    }
    expected[0] = testData[i].e;
    sum += cost();
  }
  return sum;
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



int numNodes[numlayers+1] = {3,4,3,1};

int main(){
	srand (time(NULL));
	
  network aysup;
  int i;
  for(i=0; i<=numlayers; i++){
    aysup.numNodes[i] = numNodes[i];
  }
  aysup.randomize();
  
  int a,b,c;
  int count = 0;
  for(a=-1; a<2; a++){
    for(b=-2; b<3; b++){
      for(c=-1; c<2; c++){
        testData[count] = makeSample(a,b,c,(a+b+c)*0.1);
        count++;
      }
    }
  }
  
  for(i=0; i<1000; i++){
    aysup.trainingCycle();
    //aysup.trainingInfo();
    cout<<aysup.totalCost()<<'\n';
  }
  aysup.trainingInfo();
  aysup.storeNet();
}
