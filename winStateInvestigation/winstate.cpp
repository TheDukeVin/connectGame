#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#define boardS 5
#define conn 4
#define L 1
#define trainGames 60000
#define testGames 600
using namespace std;

ofstream fout("connect.out");

struct pos{
    int x, y;
};

pos conPos(int x, int y){
    pos p;
    p.x = x;
    p.y = y;
    return p;
}

struct gamestate{
    int board[boardS][boardS];
    void startGame();
    void printState();
    void copyState(gamestate gs);
    bool winState(pos m, int team);
    pos randomMove();
    void randomGame();
    int winner;
};

void gamestate::startGame(){
    int i, j;
    for (i = 0; i < boardS; i++) {
        for (j = 0; j < boardS; j++) {
            board[i][j] = 0;
        }
    }
}

char symbs[3] = { 'O', '.', 'X' };

void gamestate::printState(){
    int i, j;
    for (i = 0; i < boardS; i++) {
        for (j = 0; j < boardS; j++) {
            cout << symbs[board[i][j] + 1];
        }
        cout << '\n';
    }
}

void gamestate::copyState(gamestate gs){
    int i, j;
    for (i = 0; i < boardS; i++) {
        for (j = 0; j < boardS; j++) {
            board[i][j] = gs.board[i][j];
        }
    }
}

int connDir[4][2] = { { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

bool gamestate::winState(pos m, int team){
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
            if (px < 0 || px >= boardS) {
                break;
            }
            if (py < 0 || py >= boardS) {
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
            if (px < 0 || px >= boardS) {
                break;
            }
            if (py < 0 || py >= boardS) {
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

pos gamestate::randomMove(){
    int spaces = 0;
    int i, j;
    for (i = 0; i < boardS; i++) {
        for (j = 0; j < boardS; j++) {
            if (board[i][j] == 0) {
                spaces++;
            }
        }
    }
    int moveID = rand() % spaces;
    int count = 0;
    for (i = 0; i < boardS; i++) {
        for (j = 0; j < boardS; j++) {
            if (board[i][j] == 0) {
                if (count == moveID) {
                    return conPos(i, j);
                }
                count++;
            }
        }
    }
    return conPos(-1, -1);
}

void gamestate::randomGame(){
    startGame();
    int i;
    int team;
    pos nextMove;
    bool won;
    for (i = 0; i < boardS * boardS; i++) {
        if (i % 2 == 0) {
            team = 1;
        }
        else {
            team = -1;
        }
        nextMove = randomMove();
        board[nextMove.x][nextMove.y] = team;
        won = winState(nextMove, team);
        if (won) {
            winner = team;
            return;
        }
    }
}

int N[L + 1] = { 25, 1 };
double mats[L][30][30];
double Av[L + 1][30];
double Bv[L][30];
double dmats[L][30][30];
double dAv[L + 1][30];
double dBv[L][30];
double dmatssum[L][30][30];

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double dsigmoid(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

double squ(double x){
    return x * x;
}

void randomize(){
    int l, i, j;
    for (l = 0; l < L; l++) {
        for (i = 0; i < N[l]; i++) {
            for (j = 0; j < N[l + 1]; j++) {
                mats[l][i][j] = 2 * (double)rand() / RAND_MAX - 1;
            }
        }
    }
}

void saveNet(){
    int l, i, j;
    for (l = 0; l < L; l++) {
        for (i = 0; i < N[l]; i++) {
            for (j = 0; j < N[l + 1]; j++) {
                fout << mats[l][i][j] << ' ';
            }
            fout << '\n';
        }
        fout << '\n';
    }
}

void pass(gamestate gs){
    int l, i, j;
    for (i = 0; i < boardS; i++) {
        for (j = 0; j < boardS; j++) {
            Av[0][i * boardS + j] = gs.board[i][j];
        }
    }
    double s;
    for (l = 0; l < L; l++) {
        for (j = 0; j < N[l + 1]; j++) {
            s = 0;
            for (i = 0; i < N[l]; i++) {
                s += Av[l][i] * mats[l][i][j];
            }
            Bv[l][j] = s;
            Av[l + 1][j] = sigmoid(s);
        }
    }
}

void backpass(gamestate gs, double expected){
    int l, i, j;
    dAv[L][0] = 2 * (Av[L][0] - expected);
    double s;
    for (l = L - 1; l >= 0; l--) {
        for (i = 0; i < N[l + 1]; i++) {
            dBv[l][i] = dAv[l + 1][i] * dsigmoid(Bv[l][i]);
        }
        for (i = 0; i < N[l]; i++) {
            s = 0;
            for (j = 0; j < N[l + 1]; j++) {
                s += dBv[l][j] * mats[l][i][j];
                dmats[l][i][j] = dBv[l][j] * Av[l][i];
            }
            dAv[l][i] = s;
        }
    }
}

void initTrain(){
    int l, i, j;
    for (l = 0; l < L; l++) {
        for (i = 0; i < N[l]; i++) {
            for (j = 0; j < N[l + 1]; j++) {
                dmatssum[l][i][j] = 0;
            }
        }
    }
}

void incdmat(){
    int l, i, j;
    for (l = 0; l < L; l++) {
        for (i = 0; i < N[l]; i++) {
            for (j = 0; j < N[l + 1]; j++) {
                dmatssum[l][i][j] += dmats[l][i][j];
            }
        }
    }
}

void learn(){
    int l, i, j;
    for (l = 0; l < L; l++) {
        for (i = 0; i < N[l]; i++) {
            for (j = 0; j < N[l + 1]; j++) {
                mats[l][i][j] -= dmatssum[l][i][j] * 0.00001;
            }
        }
    }
}

gamestate states[trainGames];
gamestate testStates[testGames];

void rollStates(){
    int i;
    for (i = 0; i < trainGames; i++) {
        states[i].randomGame();
    }
    for (i = 0; i < testGames; i++) {
        testStates[i].randomGame();
    }
}

void epoch(){
    initTrain();
    int i;
    double expected;
    for (i = 0; i < trainGames; i++) {
        expected = ((double)states[i].winner + 1) / 2;
        backpass(states[i], expected);
        incdmat();
    }
    learn();
}

double test(){
    int i;
    double expected;
    double error = 0;
    for (i = 0; i < testGames; i++) {
        expected = ((double)testStates[i].winner + 1) / 2;
        pass(testStates[i]);
        error += squ(Av[L][0] - expected);
    }
    return error;
}

int main(){
    srand(time(0));
    randomize();
    rollStates();
    int i;
    for (i = 0; i < 20; i++) {
        cout << test() << '\n';
        epoch();
    }
    for(i=0; i<5; i++){
    	testStates[i].printState();
    	pass(testStates[i]);
    	cout<<testStates[i].winner<<' '<<Av[L][0];
	}
}

