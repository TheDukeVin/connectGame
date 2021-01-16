
#include <iostream>
#include <fstream>
#define boardx 5
#define boardy 5
#define conn 4
using namespace std;

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
	int p1board[boardx][boardy];
	int p2board[boardx][boardy];
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

int main(){
	gamestate gs;
	gs.startGame();
	gs.board[0][0] = 1;
	gs.board[0][2] = 1;
	gs.board[0][3] = 1;
	cout<<gs.winState(conMove(0,1),1);
	cout<<gs.winState(conMove(0,4),1);
}
