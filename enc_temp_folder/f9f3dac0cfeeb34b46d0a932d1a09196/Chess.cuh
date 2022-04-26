#include "RNN.cuh"

class Position
{
public:

	int i;
	int j;

	Position();

	Position(int pi, int pj);

	Position(const Position& p);

	void sum(int si, int sj);
};

enum move_type {
	NORMAL, EN_PASSANT, PROMQ, PROMK, PROMR, PROMB, CASTLK, CASTLQ
};

class Move
{
public:

	Position pini;
	Position pfi;
	move_type mov_t;
	bool check = false;

	Move(Position pi, Position pf);
};

class chess_board 
{
	int board[8][8] = { {-2,-3,-4,-5,-6,-4,-3,-2 },
					    {-1,-1,-1,-1,-1,-1,-1,-1 },
						{ 0, 0, 0, 0, 0, 0, 0, 0 },
						{ 0, 0, 0, 0, 0, 0, 0, 0 },
						{ 0, 0, 0, 0, 0, 0, 0, 0 },
						{ 0, 0, 0, 0, 0, 0, 0, 0 },
						{ 1, 1, 1, 1, 1, 1, 1, 1 },
						{ 2, 3, 4, 5, 6, 4, 3, 2 }
					  };
	chess_board();

	chess_board(const chess_board& cb);

	chess_board move_piece(Move m);
};

class chess_game
{
	chess_board b;
	bool white_turn = true;

	std::vector<Move> possible_moves;

	void generate_semi_legal_moves();

	void generate_pawn_moves();
	void generate_rook_moves();
	void generate_bishop_moves();
	void generate_queen_moves();
	void generate_knight_moves();
	void generate_king_moves();

	void filter_ilegal_moves();

	void move_piece(Move m);
};

class chess_AI
{
	chess_game* cg;

	double evaluate(chess_board b);

	void play_best_move();
};

void chess_2AI_problem();