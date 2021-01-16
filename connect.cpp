
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#define boardx 5
#define boardy 5
#define conn 4
#define L 3
#define numgames 600
using namespace std;

ofstream fout ("connect.out");

struct move{
	int x,y;
};

move conMove(int x, int y){
	move m;
	m.x = x;
	m.y = y;
	return m;
}

//Gamestate structure

struct gamestate{
	int board[boardx][boardy];
	void startGame();
	void printState();
	void copyState(gamestate gs);
	bool winState(move m, int team);
};

void gamestate::startGame(){
	int i,j;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			board[i][j] = 0;
		}
	}
}

char symbs[3] = {'O','.','X'};

void gamestate::printState(){
	int i,j;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			cout<<symbs[board[i][j]+1];
		}
		cout<<'\n';
	}
}

void gamestate::copyState(gamestate gs){
	int i,j;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			board[i][j] = gs.board[i][j];
		}
	}
}

int connDir[4][2] = {{1,-1},{1,0},{1,1},{0,1}};

bool gamestate::winState(move m, int team){
	int i;
	int fConn,bConn;
	int px,py;
	for(i=0; i<4; i++){
		px = m.x;
		py = m.y;
		fConn = 0;
		while(true){
			px += connDir[i][0];
			py += connDir[i][1];
			if(px<0 || px>=boardx){
				break;
			}
			if(py<0 || py>=boardy){
				break;
			}
			if(board[px][py] != team){
				break;
			}
			fConn++;
		}
		px = m.x;
		py = m.y;
		bConn = 0;
		while(true){
			px -= connDir[i][0];
			py -= connDir[i][1];
			if(px<0 || px>=boardx){
				break;
			}
			if(py<0 || py>=boardy){
				break;
			}
			if(board[px][py] != team){
				break;
			}
			bConn++;
		}
		if(1 + fConn + bConn >= conn){
			return true;
		}
	}
	return false;
}

//Network Structure

struct network{
	int N[L+1];
	double mats[L][25][25];
	double Av[L+1][25];
	double Bv[L][25];
	double dmats[L][25][25];
	double dAv[L+1][25];
	double dBv[L][25];
	double expv[25];
	
	double dmatssum[L][25][25];
	
	void randomize();
	void saveNet();
	void pass(gamestate gs, int team);
	void backpass(gamestate gs, int team);
	
	void initTrain();
	void incdmat();
	void learn();
	
	void trainValit();
	double assessError();
	double assessError2();
	void trainVal();
	void trainPolit();
	void trainPol();
};

double sigmoid(double x){
	return 1 / (1 + exp(-x));
}

double dsigmoid(double x){
	return sigmoid(x) * (1 - sigmoid(x));
}

void network::randomize(){
	int l,i,j;
	for(l=0; l<L; l++){
		for(i=0; i<N[l]; i++){
			for(j=0; j<N[l+1]; j++){
				mats[l][i][j] = 2 * (double)rand() / RAND_MAX - 1;
			}
		}
	}
}

void network::saveNet(){
	int l,i,j;
	for(l=0; l<L; l++){
		for(i=0; i<N[l]; i++){
			for(j=0; j<N[l+1]; j++){
				fout<<mats[l][i][j]<<' ';
			}
			fout<<'\n';
		}
		fout<<'\n';
	}
}

void network::pass(gamestate gs, int team){
	int l,i,j;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			Av[0][i*boardy+j] = gs.board[i][j] * team;
		}
	}
	double s;
	for(l=0; l<L; l++){
		for(j=0; j<N[l+1]; j++){
			s = 0;
			for(i=0; i<N[l]; i++){
				s += Av[l][i] * mats[l][i][j];
			}
			Bv[l][j] = s;
			Av[l+1][j] = sigmoid(s);
		}
	}
}

void network::backpass(gamestate gs, int team){
	pass(gs,team);
	int l,i,j;
	for(i=0; i<N[L]; i++){
		dAv[L][i] = 2 * (Av[L][i] - expv[i]);
		if(expv[i] == -1){ // don't consider if expected value is -1 (null)
			dAv[L][i] = 0;
		}
	}
	double s;
	for(l=L-1; l>=0; l--){
		for(i=0; i<N[l+1]; i++){
			dBv[l][i] = dAv[l+1][i] * dsigmoid(Bv[l][i]);
		}
		for(i=0; i<N[l]; i++){
			s = 0;
			for(j=0; j<N[l+1]; j++){
				s += dBv[l][j] * mats[l][i][j];
				dmats[l][i][j] = dBv[l][j] * Av[l][i];
			}
			dAv[l][i] = s;
		}
	}
}

void network::initTrain(){
	int l,i,j;
	for(l=0; l<L; l++){
		for(i=0; i<N[l]; i++){
			for(j=0; j<N[l+1]; j++){
				dmatssum[l][i][j] = 0;
			}
		}
	}
}

void network::incdmat(){
	int l,i,j;
	for(l=0; l<L; l++){
		for(i=0; i<N[l]; i++){
			for(j=0; j<N[l+1]; j++){
				dmatssum[l][i][j] += dmats[l][i][j];
			}
		}
	}
}

void network::learn(){
	int l,i,j;
	for(l=0; l<L; l++){
		for(i=0; i<N[l]; i++){
			for(j=0; j<N[l+1]; j++){
				mats[l][i][j] -= dmatssum[l][i][j] * 0.00003;
			}
		}
	}
}






network policyNet;
network valueNet;

//Training structure

struct game{
	int moves;
	gamestate states[26];
	gamestate augStates[26][8];
	double values[26];
	double moveProbs[26][25];
	void rollGame();
	void evalStates();
	void findMoveProbs();
	void augmentStates();
};

double probs[boardx*boardy];

double squ(double x){
	return x*x;
}

move makeMove(gamestate gs, int team){
	policyNet.pass(gs,team);
	int i,j;
	double s = 0;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			probs[i*boardy+j] = squ(policyNet.Av[L][i*boardy+j]);
			if(gs.board[i][j] == 0){
				s += probs[i*boardy+j];
			}
		}
	}
	double val = (double)rand() / RAND_MAX * s;
	double parsum = 0;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			if(gs.board[i][j] == 0){
				parsum += probs[i*boardy+j];
				if(val<=parsum){
					return conMove(i,j);
				}
			}
		}
	}
}

void game::rollGame(){
	gamestate gs;
	gs.startGame();
	states[0].copyState(gs);
	int i;
	int team;
	move nextMove;
	bool won;
	for(i=0; i<boardx*boardy; i++){
		if(i%2 == 0){
			team = 1;
		}
		else{
			team = -1;
		}
		nextMove = makeMove(gs,team);
		gs.board[nextMove.x][nextMove.y] = team;
		
		states[i+1].copyState(gs);
		won = gs.winState(nextMove,team);
		if(won){
			moves = i+1;
			values[moves] = (team+1) / 2;
			return;
		}
	}
	moves = boardx*boardy;
	values[moves] = 0.5;
}

int simulate(gamestate gs, int moveIndex){
	int i;
	int team;
	move nextMove;
	bool won;
	for(i=moveIndex; i<boardx*boardy; i++){
		if(i%2 == 0){
			team = 1;
		}
		else{
			team = -1;
		}
		nextMove = makeMove(gs,team);
		gs.board[nextMove.x][nextMove.y] = team;
		
		won = gs.winState(nextMove,team);
		if(won){
			return team;
		}
	}
	return 0;
}

double evaluate(gamestate gs, int moveIndex){
	int sims;
	double sum = 0;
	int numsims = 30;
	for(sims=0; sims < numsims; sims++){
		sum += simulate(gs,moveIndex);
	}
	return (sum/numsims+1)/2;
}

void game::evalStates(){
	int i;
	for(i=0; i<moves; i++){
		values[i] = evaluate(states[i],i);
	}
}

void game::findMoveProbs(){
	int k,i,j;
	int team;
	for(k=0; k<moves; k++){
		if(k%2 == 0){
			team = 1;
		}
		else{
			team = -1;
		}
		for(i=0; i<boardx; i++){
			for(j=0; j<boardy; j++){
				if(states[k].board[i][j] != 0){
					moveProbs[k][i*boardy+j] = -1;
					continue;
				}
				states[k].board[i][j] = team;
				valueNet.pass(states[k],team);
				states[k].board[i][j] = 0;
				moveProbs[k][i*boardy+j] = valueNet.Av[L][0];
			}
		}
	}
}

void game::augmentStates(){
	int k,i,j,l;
	for(k=0; k<=moves; k++){
		for(i=0; i<boardx; i++){
			for(j=0; j<boardx; j++){
				augStates[k][0].board[i][j] = states[k].board[i][j];
				augStates[k][1].board[i][j] = states[k].board[boardx-j-1][i];
				augStates[k][2].board[i][j] = states[k].board[boardx-i-1][boardx-j-1];
				augStates[k][3].board[i][j] = states[k].board[j][boardx-i-1];
				for(l=0; l<4; l++){
					augStates[k][4+l].board[i][j] = augStates[k][l].board[i][boardx-j-1];
				}
			}
		}
	}
}



// Training algorithm

game games[numgames];

void network::trainValit(){
	initTrain();
	int i,j;
	int k;
	for(k=0; k<8; k++){
		for(i=0; i<numgames; i++){
			for(j=0; j<=games[i].moves; j++){
				if(j%2 == 0){
					expv[0] = games[i].values[j];
					backpass(games[i].augStates[j][k],1);
				}
				else{
					expv[0] = 1 - games[i].values[j];
					backpass(games[i].augStates[j][k],-1);
				}
				incdmat();
			}
		}
	}
	learn();
}

double network::assessError(){
	int i,j;
	double error = 0;
	for(i=0; i<numgames; i++){
		for(j=0; j<=games[i].moves; j++){
			if(j%2 == 0){
				expv[0] = games[i].values[j];
				pass(games[i].states[j],1);
			}
			else{
				expv[0] = 1 - games[i].values[j];
				pass(games[i].states[j],-1);
			}
			error += squ(Av[L][0] - expv[0]);
		}
	}
	return error;
}

double network::assessError2(){
	int i,j;
	double error = 0;
	int k;
	game assessGame;
	for(k=0; k<60; k++){
		assessGame.rollGame();
		assessGame.evalStates();
		for(j=0; j<=assessGame.moves; j++){
			if(j%2 == 0){
				expv[0] = assessGame.values[j];
				pass(assessGame.states[j],1);
			}
			else{
				expv[0] = 1 - assessGame.values[j];
				pass(assessGame.states[j],-1);
			}
			error += squ(Av[L][0] - expv[0]);
		}
	}
	return error;
}

void network::trainVal(){
	int i;
	for(i=0; i<numgames; i++){
		games[i].rollGame();
		games[i].evalStates();
		games[i].augmentStates();
	}
	int epoch;
	for(epoch=0; epoch<200; epoch++){
		trainValit();
		if(epoch%10 == 0){
			cout<<assessError2()<<'\n';
		}
	}
}

void network::trainPolit(){
	initTrain();
	
}

void network::trainPol(){
	int i;
	for(i=0; i<numgames; i++){
		games[i].rollGame();
		games[i].findMoveProbs();
	}
	int epoch;
	for(epoch=0; epoch<1000; epoch++){
		trainPolit();
	}
}

int main(){
	srand(time(0));
	policyNet.N[0] = 25;
	policyNet.N[1] = 30;
	policyNet.N[2] = 30;
	policyNet.N[3] = 25;
	policyNet.randomize();
	valueNet.N[0] = 25;
	valueNet.N[1] = 30;
	valueNet.N[2] = 25;
	valueNet.N[3] = 1;
	valueNet.randomize();
	
	
	valueNet.trainVal();
	valueNet.saveNet();
	
	game g;
	
	g.rollGame();
	g.evalStates();
	int i;
	for(i=0; i<=g.moves; i++){
		g.states[i].printState();
		valueNet.pass(g.states[i],1);
		cout<<g.values[i]<<' '<<valueNet.Av[L][0]<<"\n\n";
	}
	
}
