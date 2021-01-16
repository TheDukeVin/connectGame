
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <algorithm>
#define maxLayers 10
#define maxNodes 10
#define maxSpeciesPopulation 100
#define maxSpecies 100
#define boardx 3
#define boardy 3
#define conn 3
#define speciesMemory 10
#define waitTime 100
#define topNetworkPercent 0.10
#define topSpeciesPercent 0.10
#define nodeCost 0.01
using namespace std;

ofstream fout("tttspecies.out");
ifstream fin("tttspecies.in");

//Gamestate structure

struct pos {
	int x, y;
};

struct gamestate {
	int board[boardx][boardy];
	void startGame();
	void printState();
	void copyState(gamestate gs);
	bool winState(pos m, int team);
};

struct network {
	double weights[maxLayers][maxNodes][maxNodes];
	double bias[maxLayers][maxNodes];
	double activation[maxLayers + 1][maxNodes];
	int score;
	int numlayers;
	int numNodes[maxLayers + 1];

	void randomize();
	void initialize();
	void copy(network net);
	void mutate();
	void pass();
	void storeNet();
	void readNet();
	pos makemove(gamestate gs, int team);
};

struct species {
	double score[speciesMemory];
	int timer;
	int numlayers;
	int numNodes[maxLayers + 1];
	double maxScore;
	int numNets;
	network nets[maxSpeciesPopulation];
	double finalScore;
	int totalNodes;

	void updateSpecies();
	void initialize();
	void copy();
	void mutate();
	void addNode();
	void deleteNode();
	void addLayer();
	void deleteLayer();
};

bool operator < (network n1, network n2) {
	return n1.score > n2.score;
}

bool operator < (species s1, species s2) {
	return s1.finalScore > s2.finalScore;
}

void randVal(){
	return ((double)rand() / RAND_MAX)*2-1;
}

void species::updateSpecies() {
	int i;
	double sum = 0;
	double maxSum = 0;
	int top = max(5, (int)(numNets * topNetworkPercent));
	sort(nets, nets + numNets);
	for (i = 0; i < top; i++) {
		sum += nets[i].score;
	}
	sum /= top;
	for (i = 0; i < speciesMemory - 1; i++) {
		score[i] = score[i + 1];
	}
	score[speciesMemory - 1] = sum;
	
	for (i = 0; i < speciesMemory; i++) {
		maxSum += score[i];
	}
	maxSum /= speciesMemory;
	if (maxSum > maxScore) {
		maxScore = maxSum;
		timer = 0;
	}
	timer++;
	for (i = 0; i < speciesMemory - 1; i++) {
		score[i] = score[i + 1];
	}
	for (i = top; i < numNets; i++) {
		nets[i].copy(nets[i % top]);
		nets[i].mutate();
	}
}

void species::initialize() {
	int i;
	for (i = 0; i < speciesMemory; i++) {
		score[i] = 0;
	}
	timer = 0;
	maxScore = -2000000;
}

void species::copy(species spec) {
	numlayers = spec.numlayers;
	numNets = spec.numNets;
	int i;
	for(i=0; i<numlayers; i++){
		numNodes[i] = spec.numNodes[i];
	}
	for(i=0; i<spec.numNets; i++){
		nets[i].copy(spec.nets[i]);
	}
}

bool nodeOption[maxLayers+1];

void species::addNode(){
	int i;
	int count = 0;
	for(i=1; i<numlayer; i++){
		if(numNodes[i] < maxNodes){
			nodeOption[i] = true;
			count++;
		}
		else{
			nodeOption[i] = false;
		}
	}
	int randNum = rand()%count;
	count = 0;
	int changeLayer;
	for(i=1; i<numlayer; i++){
		if(numNodes[i] < maxNodes){
			if(count == randNum){
				changeLayer = i;
			}
			count++;
		}
	}
	int newNode = numNodes[changeLayer+1];
	numNodes[changeLayer+1]++;
	int n;
	for(n=0; n<numNets; i++){
		for(i=0; i<numNodes[changeLayer]; i++){
			nets[n].weights[changeLayer][i][newNode] = randVal();
		}
		for(i=0; i<numNodes[changeLayer+2]; i++){
			nets[n].weights[changeLayer+1][newNode][i] = randVal();
		}
		nets[n].bias[changeLayer][newNode] = randVal();
	}
}

void species::deleteNode(){
	int i;
	int count = 0;
	for(i=1; i<numlayer; i++){
		if(numNodes[i] > 1){
			nodeOption[i] = true;
			count++;
		}
		else{
			nodeOption[i] = false;
		}
	}
	int randNum = rand()%count;
	count = 0;
	int changeLayer;
	for(i=1; i<numlayer; i++){
		if(numNodes[i] < maxNodes){
			if(count == randNum){
				changeLayer = i;
			}
			count++;
		}
	}
	numNodes[changeLayer+1]--;
}

void species::addLayer(){
	numlayers++;
	numNodes[numlayers] = numNodes[numlayers-1];
	int n,i,j;
	for(n=0; n<numNets; n++){
		for(i=0; i<numNodes[numlayers-1]; i++){
			for(j=0; j<numNodes[numlayers]; j++){
				nets[n].weights[numlayers-1][i][j] = randVal();
			}
		}
		for(i=0; i<numNodes[numlayers]; i++){
			nets[n].bias[numlayers-1][i] = randVal();
		}
	}
}

void species::deleteLayer(){
	numlayers--;
	numNodes[numlayers] = boardx*boardy;
	int n,i,j;
	for(n=0; n<numNets; n++){
		for(i=0; i<numNodes[numlayers-1]; i++){
			for(j=0; j<numNodes[numlayers]; j++){
				nets[n].weights[numlayers-1][i][j] = randVal();
			}
		}
		for(i=0; i<numNodes[numlayers]; i++){
			nets[n].bias[numlayers-1][i] = randVal();
		}
	}
}

bool mutateChoices[4];

void species::mutate(){
	int i;
	for(i=0; i<4; i++){
		mutateChoices[i] = false;
	}
	if(numlayers < maxLayers){
		mutateChoices[0] = true;
	}
	if(numlayers > 1){
		mutateChoices[1] = true;
	}
	
}

pos conMove(int x, int y) {
	pos m;
	m.x = x;
	m.y = y;
	return m;
}

void gamestate::startGame() {
	int i, j;
	for (i = 0; i < boardx; i++) {
		for (j = 0; j < boardy; j++) {
			board[i][j] = 0;
		}
	}
}

char symbs[3] = { 'O','.','X' };

void gamestate::printState() {
	int i, j;
	for (i = 0; i < boardx; i++) {
		for (j = 0; j < boardy; j++) {
			cout << symbs[board[i][j] + 1];
		}
		cout << '\n';
	}
}

void gamestate::copyState(gamestate gs) {
	int i, j;
	for (i = 0; i < boardx; i++) {
		for (j = 0; j < boardy; j++) {
			board[i][j] = gs.board[i][j];
		}
	}
}

int connDir[4][2] = { {1,-1},{1,0},{1,1},{0,1} };

bool gamestate::winState(pos m, int team) {
	int i;
	int fConn, bConn;
	int px, py;
	for (i = 0; i < 4; i++) {
		px = m.x;
		py = m.y;
		fConn = 0;
		while (true) {
			px += connDir[i][0];
			py += connDir[i][1];
			if (px < 0 || px >= boardx) {
				break;
			}
			if (py < 0 || py >= boardy) {
				break;
			}
			if (board[px][py] != team) {
				break;
			}
			fConn++;
		}
		px = m.x;
		py = m.y;
		bConn = 0;
		while (true) {
			px -= connDir[i][0];
			py -= connDir[i][1];
			if (px < 0 || px >= boardx) {
				break;
			}
			if (py < 0 || py >= boardy) {
				break;
			}
			if (board[px][py] != team) {
				break;
			}
			bConn++;
		}
		if (1 + fConn + bConn >= conn) {
			return true;
		}
	}
	return false;
}

//Game structure

struct game {
	int moves;
	gamestate states[boardx * boardy + 1];
	int winTeam;

	void rollGame(network n1, network n2);
	void printGame();
};

pos network::makemove(gamestate gs, int team) {
	int i, j;
	for (i = 0; i < boardx; i++) {
		for (j = 0; j < boardy; j++) {
			activation[0][i * boardy + j] = gs.board[i][j];
		}
	}
	pass();
	int max = -1;
	int index;
	for (i = 0; i < boardx * boardy; i++) {
		if (activation[numlayers][i] * team > max) {
			index = i;
			max = activation[numlayers][i] * team;
		}
	}
	return conMove(index / boardy, index % boardy);
}

void game::rollGame(network n1, network n2) {
	gamestate gs;
	gs.startGame();
	states[0].copyState(gs);
	int i;
	int team;
	pos nextMove;
	bool won;
	for (i = 0; i < boardy * boardy; i++) {
		if (i % 2 == 0) {
			team = 1;
			nextMove = n1.makemove(gs, 1);
		}
		else {
			team = -1;
			nextMove = n2.makemove(gs, -1);
		}

		gs.board[nextMove.x][nextMove.y] = team;
		states[i + 1].copyState(gs);
		won = gs.winState(nextMove, team);
		if (won) {
			moves = i + 1;
			winTeam = team;
			return;
		}
	}
	moves = boardx * boardy;
	winTeam = 0;
}

int numspec = 1;
species spec[maxSpecies];
int totalNets;
network allNets[maxSpecies * maxSpeciesPopulation];

void checkSpecies() {
	int s;
	bool ready = true;
	for (s = 0; s < numspec; s++) {
		if (spec[s].timer < waitTime) {
			ready = false;
			break;
		}
	}
	if (ready) {
		evolveSpecies();
	}
}

void evolveSpecies() {
	int i, s, n;
	battleAllNets(0.5);
	int top;
	for (s = 0; s < numspec; s++) {
		top = max(1, (int)(spec[s].numNets * topNetworkPercent));
		spec[s].finalScore = 0;
		for(n = 0; n < top; n++) {
			spec[s].finalScore += spec[s].nets[n].score;
		}
		spec[s].finalScore /= top;
		spec[s].finalScore -= (double)totalNodes * nodeCost;
	}
	sort(spec, spec + numspec);
	top = max(1, (int)(numspec * topSpeciesPercent));
	for (i = top; i < numNets; i++) {
		nets[i].copy(nets[i % top]);
		nets[i].mutate();
	}
}

void combineTopNets() {
	int s, n, top;
	int count = 0;
	totalNets = 0;
	for (s = 0; s < numspec; s++) {
		top = max(1, (int)(spec[s].numNets * topNetworkPercent));
		for (n = 0; n < top; n++) {
			allNets[count] = spec[s].nets[n];
			count++;
		}
		totalNets += top;
	}
}

void combineAllNets() {
	int s, n;
	int count = 0;
	totalNets = 0;
	for (s = 0; s < numspec; s++) {
		for (n = 0; n < spec[s].numNets; n++) {
			allNets[count] = spec[s].nets[n];
			count++;
		}
		totalNets += spec[s].numNets;
	}
}

void battleAllNets(double percent) {
	int i, j;
	int other;
	int randnum;
	int battleLength = max(18, (int)(totalNets * percent));
	game g;
	network hold;
	for (i = 0; i < totalNets; i++) {
		allNets[i].score = 0;
	}
	for (i = 0; i < totalNets; i++) {
		randnum = rand() % totalNets;
		hold = allNets[i];
		allNets[i] = allNets[randnum];
		allNets[randnum] = hold;
	}
	for (i = 0; i < totalNets; i++) {
		for (j = 0; j < battleLength; j++) {
			other = (i + j) % totalNets;
			g.rollGame(allNets[i], allNets[other]);
			allNets[i].score += g.winTeam;
			allNets[other].score -= g.winTeam;
		}
	}
}

int main() {
	srand(time(NULL));
}




