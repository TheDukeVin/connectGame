
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#define boardx 3
#define boardy 3
#define N boardx*boardy
#define conn 3
#define numgames 1000
#define learnRate 0.04
using namespace std;

ofstream fout ("valuenet.out");

struct move{
	int x,y;
	void printMove();
};

move conMove(int x, int y){
	move m;
	m.x = x;
	m.y = y;
	return m;
}

void move::printMove(){
	cout<<x<<' '<<y<<'\n';
}

//Gamestate structure

struct gamestate{
	int board[boardx][boardy];
	int team; // player about to make the next move
	
	void startGame();
	void printState();
	void copyState(gamestate gs);
	bool winState(move m);
};

void gamestate::startGame(){
	int i,j;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			board[i][j] = 0;
		}
	}
	team = 1;
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
	cout<<team;
}

void gamestate::copyState(gamestate gs){
	int i,j;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			board[i][j] = gs.board[i][j];
		}
	}
	team = gs.team;
}

int connDir[4][2] = {{1,-1},{1,0},{1,1},{0,1}};

bool gamestate::winState(move m){
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
	double inp[N];
	double weight[N];
	double sum;
	
	double Dweight[N];
	double Sweight[N];
	
	void initiate();
	void saveNet();
	double pass(gamestate gs);
	move makemove(gamestate gs);
	
	double cost();
	void backProp(gamestate gs, double expected);
	void initS();
	void increment();
	void updateNet();
	void trainIteration();
	void train();
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

double invnonlinear(double x){
  return 0.5 * (log(1+x) - log(1-x));
}

void network::initiate(){
	int i;
	for(i=0; i<N; i++){
		weight[i] = 0;
	}
}

void network::saveNet(){
	int i;
	for(i=0; i<N; i++){
		fout<<weight[i]<<' ';
		if(i%boardy == boardy-1) fout<<'\n';
	}
}

double network::pass(gamestate gs){
	int i,j;
	int cell;
	for(i=0; i<N; i++){
		inp[i] = 0;
	}
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			inp[i*boardy+j] = gs.board[i][j] * gs.team;
		}
	}
	sum = 0;
	for(i=0; i<N; i++){
		sum += inp[i] * weight[i];
	}
	return nonlinear(sum);
}

double props[boardx*boardy];

double propDist(double v){
	return (1-v) / (1+v);
}

move network::makemove(gamestate gs){
	int i,j;
	double sum = 0;
	double prop;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			if(gs.board[i][j] == 0){
				gs.board[i][j] = gs.team;
				gs.team *= -1;
				prop = propDist(pass(gs));
				gs.team *= -1;
				gs.board[i][j] = 0;
				props[i*boardy+j] = prop;
				sum += prop;
			}
		}
	}
	double randVal = (double)rand() / RAND_MAX * sum;
	double parsum = 0;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			if(gs.board[i][j] == 0){
				parsum += props[i*boardy+j];
				if(randVal<=parsum){
					return conMove(i,j);
				}
			}
		}
	}
}

//Game structure

struct game{
	int moves;
	gamestate states[boardx*boardy+1];
	int winTeam;
	
	void rollGame(network n1, network n2);
	void printGame();
};

void game::rollGame(network n1, network n2){
	gamestate gs;
	gs.startGame();
	states[0].copyState(gs);
	int i;
	move nextMove;
	bool won;
	for(i=0; i<boardy*boardy; i++){
		if(i%2 == 0){
			nextMove = n1.makemove(gs);
		}
		else{
			nextMove = n2.makemove(gs);
		}
		
		won = gs.winState(nextMove);
		gs.board[nextMove.x][nextMove.y] = gs.team;
		gs.team *= -1;
		states[i+1].copyState(gs);
		if(won){
			moves = i+1;
			winTeam = -gs.team;
			return;
		}
	}
	moves = boardx*boardy;
	winTeam = 0;
}

void game::printGame(){
	int i;
	for(i=0; i<=moves; i++){
		states[i].printState();
		cout<<'\n';
	}
	cout<<winTeam;
}

// Training structure

game games[numgames];

void rollAll(network n1, network n2){
	int i;
	for(i=0; i<numgames; i++){
		if(rand()%2 == 0){
			games[i].rollGame(n1,n2);
		}
		else{
			games[i].rollGame(n2,n1);
		}
	}
}

double network::cost(){
	int g,m;
	double guess,expect;
	double sum = 0;
	for(g=0; g<numgames; g++){
		for(m=0; m<=games[g].moves; m++){
			guess = pass(games[g].states[m]);
			expect = games[g].winTeam * games[g].states[m].team;
			sum += squ(guess - expect);
		}
	}
	return sum;
}

void network::backProp(gamestate gs, double expected){
	double guess = pass(gs);
	int i;
	for(i=0; i<N; i++){
		Dweight[i] = inp[i] * dnonlinear(sum) * 2*(guess-expected);
	}
}

void network::initS(){
	int i;
	for(i=0; i<N; i++){
		Sweight[i] = 0;
	}
}

void network::increment(){
	int i;
	for(i=0; i<N; i++){
		Sweight[i] += Dweight[i];
	}
}

void network::updateNet(){
	int i;
	for(i=0; i<N; i++){
		weight[i] -= Sweight[i] / numgames * learnRate;
	}
}

void network::trainIteration(){
	initS();
	
	int g,m;
	for(g=0; g<numgames; g++){
		for(m=0; m<=games[g].moves; m++){
			backProp(games[g].states[m], games[g].winTeam * games[g].states[m].team);
			increment();
		}
	}
	updateNet();
}
/*
void network::descent(){
	int g,m,i;
	for(i=0; i<N; i++){
		change[i] = 0;
	}
	double guess,expect;
	for(g=0; g<numgames; g++){
		for(m=0; m<=games[g].moves; m++){
			guess = pass(games[g].states[m]);
			expect = games[g].winTeam * games[g].states[m].team;
			for(i=0; i<N; i++){
				//cout<<inp[i]<<' '<<dsigmoid(sum)<<' '<<2*(guess-expect)<<'\n';
				change[i] += inp[i] * dnonlinear(sum) * 2*(guess-expect);
			}
		}
	}
	for(i=0; i<N; i++){
		weight[i] -= change[i] / numgames * learnRate;
	}
}
*/
void network::train(){
	network randomPlayer;
	randomPlayer.initiate();
	rollAll(*this,randomPlayer);
	int i;
	for(i=0; i<50; i++){
		//descent();
		trainIteration();
		cout<<i<<' '<<cost()<<'\n';
	}
}

int res[3];

void match(network n1, network n2){
	int i;
	for(i=0; i<3; i++){
		res[i] = 0;
	}
	game g;
	for(i=0; i<5000; i++){
		g.rollGame(n1,n2);
		res[g.winTeam+1]++;
	}
	for(i=0; i<3; i++){
		cout<<res[i]<<' ';
	}
	cout<<'\n';
}

int main(){
	srand((unsigned)time(NULL));
	/*
	network n1;
	n1.initiate();
	n1.weight[4] = 10;
	gamestate gs;
	gs.startGame();
	n1.makemove(gs).printMove();
	*/
	/*
	network n1,n2;
	n1.initiate();
	n2.initiate();
	n2.weight[4] = 10;
	rollAll(n1,n2);
	games[0].printGame();
	*/
	
	network n1,n2;
	n1.initiate();
	n2.initiate();
	rollAll(n1,n2);
	
	network student;
	student.initiate();
	student.train();
	student.saveNet();
}
