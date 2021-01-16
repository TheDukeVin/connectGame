
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#define boardx 3
#define boardy 3
#define N 2*boardx*boardy
#define conn 3
#define numgames 1000
using namespace std;

ofstream fout ("linear.out");

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
	double inp[N];
	double weight[N];
	double sum;
	double change[N];
	
	void initiate();
	void saveNet();
	double pass(gamestate gs);
	move makemove(gamestate gs, int team);
	double error();
	void descent();
	void train();
};

double sigmoid(double x){
	return 1 / (1 + exp(-x));
}

double dsigmoid(double x){
	return sigmoid(x) * (1 - sigmoid(x));
}

double invsigmoid(double x){
	return log(x/(1-x));
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
		if(i == boardx*boardy-1) fout<<'\n';
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
			cell = gs.board[i][j];
			if(cell == 1){
				inp[i*boardy+j] = 1;
			}
			if(cell == -1){
				inp[(boardx+i)*boardy+j] = 1;
			}
		}
	}
	sum = 0;
	for(i=0; i<N; i++){
		sum += inp[i] * weight[i];
	}
	return sigmoid(sum);
}

double props[boardx*boardy];

move network::makemove(gamestate gs, int team){
	int i,j;
	double sum = 0;
	double prop;
	for(i=0; i<boardx; i++){
		for(j=0; j<boardy; j++){
			if(gs.board[i][j] == 0){
				gs.board[i][j] = team;
				prop = exp(team*invsigmoid(pass(gs)));
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
	int team;
	move nextMove;
	bool won;
	for(i=0; i<boardy*boardy; i++){
		if(i%2 == 0){
			team = 1;
			nextMove = n1.makemove(gs,1);
		}
		else{
			team = -1;
			nextMove = n2.makemove(gs,-1);
		}
		
		gs.board[nextMove.x][nextMove.y] = team;
		states[i+1].copyState(gs);
		won = gs.winState(nextMove,team);
		if(won){
			moves = i+1;
			winTeam = team;
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

double squ(double x){
	return x*x;
}

double network::error(){
	int g,m;
	double guess,expect;
	double sum = 0;
	for(g=0; g<numgames; g++){
		for(m=0; m<=games[g].moves; m++){
			guess = pass(games[g].states[m]);
			expect = ((double)games[g].winTeam+1)/2;
			sum += squ(guess - expect);
		}
	}
	return sum;
}

void network::descent(){
	int g,m,i;
	for(i=0; i<N; i++){
		change[i] = 0;
	}
	double guess,expect;
	for(g=0; g<numgames; g++){
		for(m=0; m<=games[g].moves; m++){
			guess = pass(games[g].states[m]);
			expect = ((double)games[g].winTeam+1)/2;
			for(i=0; i<N; i++){
				//cout<<inp[i]<<' '<<dsigmoid(sum)<<' '<<2*(guess-expect)<<'\n';
				change[i] += inp[i] * dsigmoid(sum) * 2*(guess-expect);
			}
		}
	}
	for(i=0; i<N; i++){
		weight[i] -= change[i] / numgames;
	}
}

void network::train(){
	network randomPlayer;
	randomPlayer.initiate();
	rollAll(*this,randomPlayer);
	int i;
	for(i=0; i<100; i++){
		descent();
		cout<<i<<' '<<error()<<'\n';
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
	network n1,n2;
	n1.initiate();
	n2.initiate();
	rollAll(n1,n2);
	network student;
	student.initiate();
	
	int i;
	for(i=0; i<5; i++){
		student.train();
		
		student.saveNet();
		fout<<'\n';
	}
	match(student,n1);
	match(n1,student);
}
