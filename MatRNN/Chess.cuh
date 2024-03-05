
#include "Prune_evolve.cuh"

class Position
{
public:

	int i;
	int j;

	Position();

	Position(int pi, int pj);

	Position(const Position& p);

	bool equals_pos(const Position& p);

	void sum(int si, int sj);

	bool is_correct();
};

enum move_type {
	NORMAL, EN_PASSANT, DOUBLE_PAWN, PROMQ, PROMN, PROMR, PROMB, CASTLK, CASTLQ
};

class Move
{
public:

	Position pini;
	Position pfi;
	move_type mov_t;
	bool check = false;

	Move();
	Move(Position pi, Position pf);
};

class chess_board 
{
public:
	Move last_move;

	bool white_turn = true;
	bool check = false;
	bool white_can_castle_k = true;
	bool black_can_castle_k = true;
	bool white_can_castle_q = true;
	bool black_can_castle_q = true;
	int fifty_moves_counter = 50;

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

	chess_board(std::string fen);

	void move_piece(Move m);

	bool same_team(Position p1, Position p2);

	std::vector<Move> generate_semi_legal_moves(bool castle_moves);

	void generate_pawn_moves(Position pos, std::vector<Move>& possible_moves);
	void generate_rook_moves(Position pos, std::vector<Move>& possible_moves);
	void generate_bishop_moves(Position pos, std::vector<Move>& possible_moves);
	void generate_queen_moves(Position pos, std::vector<Move>& possible_moves);
	void generate_knight_moves(Position pos, std::vector<Move>& possible_moves);
	void generate_king_moves(Position pos, std::vector<Move>& possible_moves, bool castle_moves);
	bool no_pieces_in_between(Position pos, int jump);

	bool is_legal(Move& m);
	void filter_ilegal_moves(std::vector<Move>& possible_moves);

	std::vector<double> board_to_ai_inp();

	void print_board();
};

class chess_AI
{
public:

	int count = 0;
	int max_boards = 1000000;
	int max_depth = 3;

	cusparseHandle_t* handle;

	const int pawns = 0;
	const int rooks = 64 * 1;
	const int knights = 64 * 2;
	const int bishops = 64 * 3;
	const int queens = 64 * 4;
	const int kings = 64 * 5;
	const int total = 64 * 6;

	std::vector<double> input;

	Prune_evolver p_rnn;

	chess_AI(cusparseHandle_t& handle);
	~chess_AI();

	double evaluate(const chess_board &b, Move& m);

	double evaluate_material(const chess_board& b, Move& m);

	double evaluate_train_material(const chess_board& b, Move& m, int white, double alfa, double beta);

	double evaluate_train_itself(const chess_board& b, Move& m, int white, double alfa, double beta);

	void play_best_move(chess_board& b, int depth, int eval);

	double negamax(const chess_board& b, Move m, int white, int depth, double alfa, double beta, int eval);

};

void chess_2AI_problem();

void test_fen();

void test_board();